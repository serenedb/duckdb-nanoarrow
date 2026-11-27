#include "ipc/codecs.hpp"

#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/formats/block_codecs.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "nanoarrow_errors.hpp"

#include <mutex>

namespace duckdb {
namespace ext_nanoarrow {

static ArrowIpcCompressionType ParseArrowIpcCompressionType(const string& name) {
  // Accept the spellings DuckDB users know from other COPY formats on top of nanoarrow's
  auto lname = StringUtil::Lower(name);
  if (lname == "uncompressed") {
    lname = "none";
  } else if (lname == "lz4_frame") {
    lname = "lz4";
  }

  ArrowIpcCompressionType type;
  ArrowError error{};
  if (ArrowIpcCompressionTypeFromString(lname.c_str(), &type, &error) != NANOARROW_OK) {
    throw BinderException(
        "Unsupported compression type \"%s\" for Arrow IPC, expected one of "
        "'uncompressed', 'zstd' or 'lz4'",
        name);
  }
  return type;
}

bool ArrowIpcCompressionOptions::TrySetOption(const string& name, const Value& value) {
  if (StringUtil::CIEquals(name, "compression")) {
    type = ParseArrowIpcCompressionType(value.ToString());
  } else if (StringUtil::CIEquals(name, "compression_level")) {
    level = value.GetValue<int64_t>();
    level_set = true;
  } else {
    return false;
  }
  return true;
}

void ArrowIpcCompressionOptions::Validate() const {
  if (!level_set) {
    return;
  }
  if (type == NANOARROW_IPC_COMPRESSION_TYPE_NONE) {
    throw BinderException(
        "COMPRESSION_LEVEL requires COMPRESSION to be set to 'zstd' or 'lz4'");
  }

  // nanoarrow checks the level too but only once the encoder exists, binding is earlier
  int min_level;
  int max_level;
  NANOARROW_THROW_NOT_OK(ArrowIpcGetCompressionLevelRange(type, &min_level, &max_level));
  if (level < min_level || level > max_level) {
    throw BinderException("Compression level for %s must be between %d and %d",
                          ArrowIpcCompressionTypeToString(type), min_level, max_level);
  }
}

namespace {

bool TryGetBlockCodec(ArrowIpcCompressionType type, BlockCodec& codec) {
  switch (type) {
    case NANOARROW_IPC_COMPRESSION_TYPE_ZSTD:
      codec = BlockCodec::ZSTD;
      return true;
    case NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME:
      codec = BlockCodec::LZ4_FRAME;
      return true;
    default:
      return false;
  }
}

ArrowErrorCode BlockDecompressAdd(ArrowIpcDecompressor* decompressor,
                                  ArrowIpcCompressionType type, ArrowBufferView src,
                                  uint8_t* dst, int64_t dst_size, ArrowError* error) {
  BlockCodec codec;
  if (!TryGetBlockCodec(type, codec)) {
    ArrowErrorSet(error, "Arrow IPC compression %s is not supported",
                  ArrowIpcCompressionTypeToString(type));
    return ENOTSUP;
  }
  try {
    auto& block = *static_cast<BlockDecompressor*>(decompressor->private_data);
    string message;
    if (block.TryDecompress(codec, src.data.as_uint8, NumericCast<idx_t>(src.size_bytes),
                            dst, NumericCast<idx_t>(dst_size), message)) {
      return NANOARROW_OK;
    }
    ArrowErrorSet(error, "Failed to decompress an Arrow IPC buffer with %s: %s",
                  ArrowIpcCompressionTypeToString(type), message.c_str());
  } catch (std::exception& ex) {
    ArrowErrorSet(error, "%s", ErrorData(ex).RawMessage().c_str());
  }
  return EIO;
}

ArrowErrorCode BlockDecompressWait(ArrowIpcDecompressor*, int64_t, ArrowError*) {
  return NANOARROW_OK;
}

ArrowErrorCode SizesOnlyDecompressAdd(ArrowIpcDecompressor*, ArrowIpcCompressionType,
                                      ArrowBufferView, uint8_t*, int64_t, ArrowError*) {
  return NANOARROW_OK;
}

void SizesOnlyDecompressRelease(ArrowIpcDecompressor* decompressor) {
  decompressor->release = nullptr;
}

void BlockDecompressRelease(ArrowIpcDecompressor* decompressor) {
  delete static_cast<BlockDecompressor*>(decompressor->private_data);
  decompressor->release = nullptr;
}

constexpr idx_t kParallelCompressBytes = 256 * 1024;

class ParallelBlockCompressor {
 public:
  ParallelBlockCompressor(optional_ptr<ClientContext> context, BlockCodec codec,
                          int level)
      : context(context), codec(codec), level(level), local(codec, level) {}

  ~ParallelBlockCompressor() {
    if (executor) {
      executor->Join();
    }
  }

  void Add(ArrowBufferView src, ArrowBuffer* dst) {
    const auto size = NumericCast<idx_t>(src.size_bytes);
    const auto bound = local.MaxCompressedSize(size);
    NANOARROW_THROW_NOT_OK(ArrowBufferReserve(dst, NumericCast<int64_t>(bound)));
    if (!context || size < kParallelCompressBytes) {
      Compress(local, src, dst, bound);
      return;
    }
    if (!executor) {
      executor = make_uniq<TaskExecutor>(*context);
    }
    executor->ScheduleTask(
        [this, src, dst, bound]() {
          auto compressor = Take();
          Compress(*compressor, src, dst, bound);
          Return(std::move(compressor));
        },
        "ArrowCompressTask");
  }

  void Wait() {
    if (!executor) {
      return;
    }
    auto running = std::move(executor);
    running->WorkOnTasks();
  }

 private:
  static void Compress(BlockCompressor& compressor, ArrowBufferView src, ArrowBuffer* dst,
                       idx_t bound) {
    dst->size_bytes += NumericCast<int64_t>(
        compressor.Compress(src.data.as_uint8, NumericCast<idx_t>(src.size_bytes),
                            dst->data + dst->size_bytes, bound));
  }

  unique_ptr<BlockCompressor> Take() {
    {
      lock_guard<mutex> guard(lock);
      if (!idle.empty()) {
        auto compressor = std::move(idle.back());
        idle.pop_back();
        return compressor;
      }
    }
    return make_uniq<BlockCompressor>(codec, level);
  }

  void Return(unique_ptr<BlockCompressor> compressor) {
    lock_guard<mutex> guard(lock);
    idle.push_back(std::move(compressor));
  }

  optional_ptr<ClientContext> context;
  const BlockCodec codec;
  const int level;
  BlockCompressor local;
  unique_ptr<TaskExecutor> executor;
  mutex lock;
  vector<unique_ptr<BlockCompressor>> idle;
};

ArrowErrorCode BlockCompressAdd(ArrowIpcCompressor* compressor, ArrowBufferView src,
                                ArrowBuffer* dst, ArrowError* error) {
  try {
    static_cast<ParallelBlockCompressor*>(compressor->private_data)->Add(src, dst);
    return NANOARROW_OK;
  } catch (std::exception& ex) {
    ArrowErrorSet(error, "%s", ErrorData(ex).RawMessage().c_str());
    return EIO;
  }
}

ArrowErrorCode BlockCompressWait(ArrowIpcCompressor* compressor, int64_t,
                                 ArrowError* error) {
  try {
    static_cast<ParallelBlockCompressor*>(compressor->private_data)->Wait();
    return NANOARROW_OK;
  } catch (std::exception& ex) {
    ArrowErrorSet(error, "%s", ErrorData(ex).RawMessage().c_str());
    return EIO;
  }
}

void BlockCompressRelease(ArrowIpcCompressor* compressor) {
  delete static_cast<ParallelBlockCompressor*>(compressor->private_data);
  compressor->release = nullptr;
}

}  // namespace

nanoarrow::ipc::UniqueDecoder NewDuckDBArrowDecoder() {
  nanoarrow::ipc::UniqueDecoder decoder;
  NANOARROW_THROW_NOT_OK(ArrowIpcDecoderInit(decoder.get()));
  ArrowIpcDecompressor decompressor;
  decompressor.decompress_add = BlockDecompressAdd;
  decompressor.decompress_wait = BlockDecompressWait;
  decompressor.release = BlockDecompressRelease;
  decompressor.private_data = new BlockDecompressor();
  NANOARROW_THROW_NOT_OK(ArrowIpcDecoderSetDecompressor(decoder.get(), &decompressor));
  return decoder;
}

void SetArrowIpcDecoderSizesOnly(ArrowIpcDecoder& decoder) {
  ArrowIpcDecompressor decompressor;
  decompressor.decompress_add = SizesOnlyDecompressAdd;
  decompressor.decompress_wait = BlockDecompressWait;
  decompressor.release = SizesOnlyDecompressRelease;
  decompressor.private_data = nullptr;
  NANOARROW_THROW_NOT_OK(ArrowIpcDecoderSetDecompressor(&decoder, &decompressor));
}

void SetArrowIpcEncoderCompression(ArrowIpcEncoder& encoder,
                                   const ArrowIpcCompressionOptions& options,
                                   optional_ptr<ClientContext> context) {
  BlockCodec codec;
  if (!TryGetBlockCodec(options.type, codec)) {
    return;
  }
  ArrowIpcCompressor compressor;
  compressor.compression_type = options.type;
  compressor.compress_add = BlockCompressAdd;
  compressor.compress_wait = BlockCompressWait;
  compressor.release = BlockCompressRelease;
  compressor.private_data =
      new ParallelBlockCompressor(context, codec,
                                  options.level_set ? NumericCast<int>(options.level)
                                                    : BlockCompressor::DEFAULT_LEVEL);
  NANOARROW_THROW_NOT_OK(ArrowIpcEncoderSetCompressor(&encoder, &compressor));
}

}  // namespace ext_nanoarrow
}  // namespace duckdb

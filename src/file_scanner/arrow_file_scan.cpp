#include <algorithm>

#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/parallel/callback_async_task.hpp"

#include "file_scanner/arrow_file_scan.hpp"
#include "file_scanner/arrow_multi_file_info.hpp"
#include "ipc/stream_reader/ipc_file_stream_reader.hpp"

namespace duckdb {
namespace ext_nanoarrow {

namespace {
//! Claims below this many body bytes would spend more on setup than on decoding
constexpr idx_t kMinClaimBodyBytes = 1024 * 1024;
//! A remote claim is fetched with few requests, so it holds more to keep them large
constexpr idx_t kRemoteMinClaimBodyBytes = 16 * 1024 * 1024;
atomic<idx_t> next_scan_id{1};

//! Names children as DuckDB names columns, which the multi file reader maps by name
void DeduplicateChildNames(ArrowSchema& schema) {
  vector<string> names;
  for (int64_t i = 0; i < schema.n_children; i++) {
    const auto name = schema.children[i]->name;
    names.push_back(name && name[0] ? name : "v" + to_string(i));
  }
  QueryResult::DeduplicateColumns(names);
  for (int64_t i = 0; i < schema.n_children; i++) {
    auto& child = *schema.children[i];
    if (!child.name || names[i] != child.name) {
      NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(&child, names[i].c_str()));
    }
    DeduplicateChildNames(child);
  }
  if (schema.dictionary) {
    DeduplicateChildNames(*schema.dictionary);
  }
}

std::pair<idx_t, idx_t> PartRows(idx_t length, idx_t part, idx_t parts) {
  const auto rows = AlignValue<idx_t, STANDARD_VECTOR_SIZE>((length + parts - 1) / parts);
  const auto begin = MinValue<idx_t>(rows * part, length);
  return {begin, MinValue<idx_t>(begin + rows, length)};
}

struct BatchSlice {
  shared_ptr<ArrowArrayWrapper> batch;
  vector<ArrowArray> children;
  vector<ArrowArray*> child_pointers;
};

void ReleaseSliceChild(ArrowArray* array) { array->release = nullptr; }

void ReleaseSlice(ArrowArray* array) {
  delete static_cast<BatchSlice*>(array->private_data);
  array->release = nullptr;
}

void SliceBatch(shared_ptr<ArrowArrayWrapper> batch, idx_t begin, idx_t end,
                ArrowArray& out) {
  const auto& source = batch->arrow_array;
  auto slice = make_uniq<BatchSlice>();
  slice->children.resize(static_cast<idx_t>(source.n_children));
  for (idx_t i = 0; i < slice->children.size(); i++) {
    auto& child = slice->children[i];
    child = *source.children[i];
    child.offset += source.offset + static_cast<int64_t>(begin);
    child.length = static_cast<int64_t>(end - begin);
    if (child.null_count != 0) {
      child.null_count = -1;
    }
    child.release = ReleaseSliceChild;
    slice->child_pointers.push_back(&child);
  }
  out = source;
  out.length = static_cast<int64_t>(end - begin);
  out.offset = 0;
  out.null_count = 0;
  out.children = slice->child_pointers.data();
  out.release = ReleaseSlice;
  slice->batch = std::move(batch);
  out.private_data = slice.release();
}
}  // namespace

ArrowFileScan::ArrowFileScan(ClientContext& context, const OpenFileInfo& file)
    : BaseFileReader(file), scan_id(next_scan_id++) {
  factory = make_shared_ptr<FileIPCStreamFactory>(context, file);

  factory->InitReader();
  auto& reader = static_cast<IPCFileStreamReader&>(*factory->reader);
  // A remote footer carries the schema, so reading it first skips the start of the file
  const bool has_footer = reader.TryReadFooter();
  factory->GetFileSchema(schema_root);
  DeduplicateChildNames(schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(context, arrow_table,
                                               schema_root.arrow_schema);
  names = arrow_table.GetNames();
  types = arrow_table.GetTypes();
  if (types.empty()) {
    throw InvalidInputException("Provided table/dataframe must have at least one column");
  }
  columns = MultiFileColumnDefinition::ColumnsFromNamesAndTypes(
      StringsToIdentifiers(names), types);

  // Scans move this reader away, so progress and estimates keep their own copies
  file_size = reader.FileSize();
  reader.TrackProgress(progress_offset);
  count_without_bodies = reader.CanCountWithoutBodies();
  if (has_footer || reader.TryIndexStream()) {
    factory->mapping = reader.MapFile(context);
    PlanClaims(reader, reader.IsRemote() ? kRemoteMinClaimBodyBytes : kMinClaimBodyBytes,
               TaskScheduler::QueryThreads(context));
    dictionary_blocks = reader.DictionaryBlocks();
    indexed_stream = !has_footer;
    const auto window = has_footer ? reader.FooterWindow() : reader.SchemaWindow();
    footer_window = factory->allocator.Allocate(static_cast<idx_t>(window.size_bytes));
    std::memcpy(footer_window.get(), window.data.data,
                static_cast<size_t>(window.size_bytes));
  } else if (reader.ClaimsStream()) {
    stream_claims = true;
    stream_threads = TaskScheduler::QueryThreads(context);
    indexed_stream = true;
    const auto window = reader.MessageWindow();
    footer_window = factory->allocator.Allocate(static_cast<idx_t>(window.size_bytes));
    std::memcpy(footer_window.get(), window.data.data,
                static_cast<size_t>(window.size_bytes));
  }
}

void ArrowFileScan::LoadIndex(IPCFileStreamReader& reader) {
  const ArrowBufferView window{{footer_window.get()},
                               static_cast<int64_t>(footer_window.GetSize())};
  if (indexed_stream) {
    reader.LoadSchemaMessage(window);
  } else {
    reader.LoadFooter(window);
  }
}

void ArrowFileScan::PlanClaims(IPCFileStreamReader& reader, idx_t min_claim_bytes,
                               idx_t threads) {
  const auto& file_blocks = reader.RecordBatchBlocks();
  idx_t begin = 0;
  idx_t body_bytes = 0;
  for (idx_t i = 0; i < file_blocks.size(); i++) {
    const auto block_bytes = static_cast<idx_t>(file_blocks[i].body_length);
    if (block_bytes >= 2 * min_claim_bytes && begin < i) {
      claims.push_back(BlockRange{begin, i});
      begin = i;
      body_bytes = 0;
    }
    body_bytes += block_bytes;
    if (body_bytes >= min_claim_bytes) {
      claims.push_back(BlockRange{begin, i + 1});
      begin = i + 1;
      body_bytes = 0;
    }
  }
  if (begin < file_blocks.size()) {
    claims.push_back(BlockRange{begin, file_blocks.size()});
  }
  // Even one claim reads its blocks with few large reads instead of following the stream
  blocks = file_blocks;
  if (!factory->mapping || claims.empty() || claims.size() >= threads) {
    return;
  }
  const auto parts_per_claim = threads / claims.size();
  vector<BlockRange> split;
  for (auto& claim : claims) {
    const auto parts = MinValue<idx_t>(
        parts_per_claim,
        static_cast<idx_t>(blocks[claim.begin].body_length) / min_claim_bytes);
    if (claim.end != claim.begin + 1 || parts < 2 ||
        !reader.PlainRecordBatch(blocks[claim.begin])) {
      split.push_back(std::move(claim));
      continue;
    }
    auto shared = make_shared_ptr<SharedBatch>();
    for (idx_t part = 0; part < parts; part++) {
      split.push_back(BlockRange{claim.begin, claim.end, part, parts, shared});
    }
  }
  claims = std::move(split);
}

void ArrowFileScan::ReadBatchPart(ArrowFileLocalState& lstate, ArrowArray& out) {
  const auto& claim = claims[lstate.claim_index];
  auto& shared = *claim.shared;
  shared_ptr<ArrowArrayWrapper> batch;
  {
    lock_guard<mutex> guard(shared.lock);
    if (!shared.batch) {
      auto decoded = make_shared_ptr<ArrowArrayWrapper>();
      if (!lstate.block_reader->GetNextBatch(&decoded->arrow_array)) {
        throw IOException(
            "Arrow IPC file \"%s\" ended before a record batch its footer names",
            GetFileName());
      }
      shared.batch = std::move(decoded);
    }
    batch = shared.batch;
    if (++shared.taken == claim.parts) {
      shared.batch.reset();
    }
  }
  const auto rows =
      PartRows(static_cast<idx_t>(batch->arrow_array.length), claim.part, claim.parts);
  nanoarrow::UniqueArray slice;
  SliceBatch(std::move(batch), rows.first, rows.second, *slice.get());
  nanoarrow::UniqueArrayView view;
  ArrowError error{};
  THROW_NOT_OK(IOException, &error,
               ArrowArrayViewInitFromSchema(
                   view.get(), lstate.block_reader->GetOutputSchema(), &error));
  THROW_NOT_OK(IOException, &error,
               ArrowArrayViewSetArray(view.get(), slice.get(), &error));
  THROW_NOT_OK(
      IOException, &error,
      ArrowArrayViewValidate(view.get(), NANOARROW_VALIDATION_LEVEL_FULL, &error));
  ArrowArrayMove(slice.get(), &out);
}

idx_t ArrowFileScan::ClaimCount() const {
  if (stream_claims) {
    return stream_threads;
  }
  return MaxValue<idx_t>(claims.size(), 1);
}

bool ArrowFileScan::HasFooter() const { return !claims.empty(); }

string ArrowFileScan::GetReaderType() const { return "ARROW"; }

const vector<string>& ArrowFileScan::GetNames() { return names; }
const vector<LogicalType>& ArrowFileScan::GetTypes() { return types; }

bool ArrowFileScan::TryInitializeScan(ClientContext& context,
                                      GlobalTableFunctionState& gstate_p,
                                      LocalTableFunctionState& lstate_p) {
  if (StreamClaims()) {
    auto& lstate = lstate_p.Cast<ArrowFileLocalState>();
    auto& reader = static_cast<IPCFileStreamReader&>(*factory->reader);
    if (stream_done || !reader.ReadMessages(kMinClaimBodyBytes, lstate.stream_messages)) {
      stream_done = true;
      return false;
    }
    lstate.claim_index = next_claim++;
    return true;
  }
  if (!claims.empty()) {
    // Claims go out in file order under the multi file lock, which keeps batch order
    if (next_claim >= claims.size()) {
      return false;
    }
    lstate_p.Cast<ArrowFileLocalState>().claim_index = next_claim++;
    return true;
  }
  auto& gstate = gstate_p.Cast<ArrowFileGlobalState>();
  if (gstate.files.find(file_list_idx.GetIndex()) != gstate.files.end()) {
    // Without a footer the batches can only be found by reading the stream in order
    return false;
  }
  gstate.files.insert(file_list_idx.GetIndex());
  return true;
}

class ArrowFileScan::BatchPartStream {
 public:
  BatchPartStream(ArrowFileScan& scan, ArrowFileLocalState& lstate)
      : scan(scan), lstate(lstate), stream(*lstate.block_reader) {}

  int GetSchema(ArrowSchema* schema) { return stream.GetSchema(schema); }

  int GetNext(ArrowArray* array) {
    return stream.Wrap([&]() {
      array->release = nullptr;
      if (!done) {
        done = true;
        scan.ReadBatchPart(lstate, *array);
      }
    });
  }

  const char* GetLastError() const { return stream.GetLastError(); }

 private:
  ArrowFileScan& scan;
  ArrowFileLocalState& lstate;
  IpcArrayStream stream;
  bool done = false;
};

//! The scan factory of the blocks a local state claimed. The state owns the scan data
//! that holds this factory, so the state outlives it
class ArrowFileScan::BlockScanFactory : public ArrowScanFactory {
 public:
  BlockScanFactory(ArrowFileScan& scan, ArrowFileLocalState& lstate)
      : scan(scan), lstate(lstate) {}

  void GetSchema(ArrowSchema& schema) override {
    NANOARROW_THROW_NOT_OK(ArrowSchemaDeepCopy(&scan.schema_root.arrow_schema, &schema));
  }

  unique_ptr<ArrowArrayStreamWrapper> ProduceStream(ArrowStreamParameters&) override {
    // PrepareScan already gave the reader the projection these parameters carry
    auto out = make_uniq<ArrowArrayStreamWrapper>();
    if (!scan.claims.empty() && scan.claims[lstate.claim_index].shared) {
      nanoarrow::ArrayStreamFactory<BatchPartStream>::InitArrayStream(
          new BatchPartStream(scan, lstate), &out->arrow_array_stream);
    } else {
      IpcArrayStream(*lstate.block_reader).ToArrayStream(&out->arrow_array_stream);
    }
    return out;
  }

 private:
  ArrowFileScan& scan;
  ArrowFileLocalState& lstate;
};

void ArrowFileScan::InitializeScanData(ArrowFileLocalState& lstate,
                                       shared_ptr<ArrowScanFactory> producer) {
  lstate.local_arrow_function_data =
      make_uniq<ArrowScanFunctionData>(std::move(producer));
  // A memberwise copy shares the release pointer, so a second scan would free it twice
  NANOARROW_THROW_NOT_OK(
      ArrowSchemaDeepCopy(&schema_root.arrow_schema,
                          &lstate.local_arrow_function_data->schema_root.arrow_schema));
  lstate.local_arrow_function_data->arrow_table = arrow_table;
  lstate.local_arrow_function_data->evaluate_filters = true;
  // Global column indexes and projection ids do not address this file's schema
  auto local_column_indexes = column_indexes;
  bool reads_file_column = false;
  for (idx_t i = 0; i < local_column_indexes.size(); i++) {
    if (file_row_number.IsValid() && file_row_number.GetIndex() == i) {
      local_column_indexes[i] = ColumnIndex(COLUMN_IDENTIFIER_ROW_NUMBER);
    } else {
      reads_file_column = true;
    }
  }
  if (!reads_file_column) {
    // Only virtual or constant columns are read, so any file column yields the rows
    local_column_indexes.emplace_back(0);
  }
  const vector<idx_t> no_projection_ids;
  lstate.init_input = make_uniq<TableFunctionInitInput>(*lstate.local_arrow_function_data,
                                                        std::move(local_column_indexes),
                                                        no_projection_ids, filters);
  if (!expression_map.empty()) {
    lstate.file_input.expression_map = expression_map;
    lstate.init_input->multi_file_input = lstate.file_input;
  }
}

void ArrowFileScan::OpenBlockReader(ArrowFileLocalState& lstate, bool read_columns) {
  // The previous scan borrows the previous reader, so it goes first
  lstate.table_function_input.reset();
  lstate.local_arrow_local_state.reset();
  lstate.local_arrow_global_state.reset();
  lstate.block_scan_id = 0;
  lstate.block_reader = factory->OpenReader();
  LoadIndex(*lstate.block_reader);
  InitializeScanData(lstate, make_shared_ptr<BlockScanFactory>(*this, lstate));
  // The fetch needs the projection before the scan that would push it starts
  if (read_columns) {
    vector<idx_t> projection;
    for (const auto column_id : lstate.init_input->column_ids) {
      if (!IsVirtualColumn(column_id)) {
        projection.push_back(column_id);
      }
    }
    if (!projection.empty()) {
      lstate.block_reader->SetColumnProjection(projection);
    }
    // Claim readers share one set, since a file cannot replace its dictionaries
    if (lstate.block_reader->NeedsDictionaries()) {
      lock_guard<mutex> guard(dictionary_lock);
      if (dictionaries) {
        lstate.block_reader->ShareDictionaries(dictionaries);
      } else {
        lstate.block_reader->LoadDictionaries(dictionary_blocks);
        // Shared buffers count their readers atomically only when nanoarrow has atomics
        if (ArrowSharedBufferIsThreadSafe()) {
          dictionaries = lstate.block_reader->Dictionaries();
        }
      }
    }
  }
  lstate.block_scan_id = scan_id;
}

void ArrowFileScan::StartScan(ClientContext& context, ArrowFileLocalState& lstate) {
  lstate.local_arrow_global_state =
      ArrowTableFunction::ArrowScanInitGlobal(context, *lstate.init_input);
  if (file_row_number.IsValid() && !claims.empty()) {
    const auto& claim = claims[lstate.claim_index];
    const auto& starts = BatchStarts();
    auto row = starts[claim.begin];
    if (claim.shared) {
      row += PartRows(starts[claim.end] - starts[claim.begin], claim.part, claim.parts)
                 .first;
    }
    lstate.local_arrow_global_state->Cast<ArrowScanGlobalState>().next_row_number = row;
  }
  lstate.local_arrow_local_state = ArrowTableFunction::ArrowScanInitLocalInternal(
      context, *lstate.init_input, lstate.local_arrow_global_state.get());
  lstate.table_function_input = make_uniq<TableFunctionInput>(
      lstate.local_arrow_function_data.get(), lstate.local_arrow_local_state.get(),
      lstate.local_arrow_global_state.get());
}

void ArrowFileScan::PrepareScan(ClientContext& context,
                                GlobalTableFunctionState& gstate_p,
                                LocalTableFunctionState& lstate_p) {
  // Reading waits for Scan, so ScheduleIO can fetch the claim on the async pool first
  auto& lstate = lstate_p.Cast<ArrowFileLocalState>();
  lstate.scan_started = false;
  lstate.count_reader = nullptr;
  lstate.count_owned_reader.reset();
  lstate.count_rows_left = 0;
  // Only virtual or constant columns are read, so the headers alone give the rows
  const bool count_only = column_indexes.empty() && count_without_bodies;
  if (StreamClaims()) {
    if (lstate.block_scan_id != scan_id) {
      OpenBlockReader(lstate, !count_only);
    }
    lstate.block_reader->SetMessages(std::move(lstate.stream_messages));
    if (count_only) {
      lstate.block_reader->CountOnly();
      lstate.count_reader = lstate.block_reader.get();
    }
    return;
  }
  if (claims.empty()) {
    lstate.block_scan_id = 0;
    if (count_only) {
      if (!factory->reader) {
        throw InternalException("ArrowFileScan counted a file whose reader was moved");
      }
      lstate.count_owned_reader.reset(
          static_cast<IPCFileStreamReader*>(factory->reader.release()));
      lstate.count_reader = lstate.count_owned_reader.get();
      return;
    }
    InitializeScanData(lstate, factory);
    return;
  }
  // A state keeps its reader for the next claim of the same file
  if (lstate.block_scan_id != scan_id) {
    OpenBlockReader(lstate, !count_only);
  }
  const auto& claim = claims[lstate.claim_index];
  const auto end = count_only && claim.part != 0 ? claim.begin : claim.end;
  lstate.block_reader->SetBlocks(blocks.data() + claim.begin, blocks.data() + end);
  lstate.block_reader->SetValidation(claim.shared ? NANOARROW_VALIDATION_LEVEL_DEFAULT
                                                  : NANOARROW_VALIDATION_LEVEL_FULL);
  if (count_only) {
    lstate.block_reader->CountOnly();
    lstate.count_reader = lstate.block_reader.get();
  }
}

AsyncResult ArrowFileScan::ScheduleIO(ClientContext& context,
                                      GlobalTableFunctionState& gstate_p,
                                      LocalTableFunctionState& lstate_p) {
  auto& lstate = lstate_p.Cast<ArrowFileLocalState>();
  // Without a footer the reads follow the stream, so they cannot be planned ahead
  if (claims.empty() || claims[lstate.claim_index].shared) {
    return SourceResultType::HAVE_MORE_OUTPUT;
  }
  auto& reader = *lstate.block_reader;
  vector<unique_ptr<AsyncTask>> io_tasks;
  io_tasks.push_back(make_uniq<CallbackAsyncTask>([&reader] { reader.FetchBlocks(); },
                                                  reader.BlockBytes()));
  return AsyncResult::FromTasks(std::move(io_tasks), TaskSchedulerType::ASYNC);
}

AsyncResult ArrowFileScan::Scan(ClientContext& context,
                                GlobalTableFunctionState& global_state,
                                LocalTableFunctionState& local_state, DataChunk& chunk) {
  auto& lstate = local_state.Cast<ArrowFileLocalState>();
  if (!lstate.scan_started) {
    lstate.scan_started = true;
    if (!lstate.count_reader) {
      StartScan(context, lstate);
    }
  }
  if (lstate.count_reader) {
    while (lstate.count_rows_left == 0) {
      if (!lstate.count_reader->NextBatchLength(lstate.count_rows_left)) {
        FinishClaim(lstate);
        return SourceResultType::FINISHED;
      }
    }
    const auto count = MinValue<idx_t>(lstate.count_rows_left, STANDARD_VECTOR_SIZE);
    chunk.SetChildCardinality(count);
    lstate.count_rows_left -= count;
    return SourceResultType::HAVE_MORE_OUTPUT;
  }
  ArrowTableFunction::ArrowScanFunction(context, *lstate.table_function_input, chunk);
  if (chunk.size() == 0) {
    FinishClaim(lstate);
    return SourceResultType::FINISHED;
  }
  return SourceResultType::HAVE_MORE_OUTPUT;
}

void ArrowFileScan::FinishClaim(ArrowFileLocalState& lstate) {
  // A finished state can wait in a pool, so the last batch it scanned is freed here
  lstate.table_function_input.reset();
  lstate.local_arrow_local_state.reset();
  lstate.local_arrow_global_state.reset();
  lstate.count_reader = nullptr;
  lstate.count_owned_reader.reset();
}

void ArrowFileScan::AddVirtualColumn(column_t virtual_column_id) {
  if (virtual_column_id != MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER) {
    throw InternalException("ArrowFileScan does not support virtual column %llu",
                            virtual_column_id);
  }
  file_row_number = column_ids.size() - 1;
}

const vector<idx_t>& ArrowFileScan::BatchStarts() {
  lock_guard<mutex> guard(batch_starts_lock);
  if (!batch_starts.empty()) {
    return batch_starts;
  }
  auto reader = factory->OpenReader();
  LoadIndex(*reader);
  vector<idx_t> starts{0};
  if (count_without_bodies) {
    reader->SetBlocks(blocks.data(), blocks.data() + blocks.size());
    idx_t length;
    while (reader->NextBatchLength(length)) {
      starts.push_back(starts.back() + length);
    }
  } else {
    reader->SetColumnProjection({0});
    reader->LoadDictionaries(dictionary_blocks);
    reader->SetBlocks(blocks.data(), blocks.data() + blocks.size());
    nanoarrow::UniqueArray batch;
    while (reader->GetNextBatch(batch.get())) {
      starts.push_back(starts.back() + static_cast<idx_t>(batch->length));
      batch.reset();
    }
  }
  if (starts.size() != blocks.size() + 1) {
    throw IOException(
        "Arrow IPC file \"%s\" has %llu record batches, its footer names %llu",
        GetFileName(), starts.size() - 1, blocks.size());
  }
  batch_starts = std::move(starts);
  return batch_starts;
}

void ArrowFileScan::PrepareLookup(ClientContext& context, ArrowFileLocalState& lstate,
                                  const DataChunk& chunk) {
  if (claims.empty()) {
    lstate.table_function_input.reset();
    lstate.local_arrow_local_state.reset();
    lstate.local_arrow_global_state.reset();
    lstate.block_reader.reset();
    InitializeScanData(lstate, factory);
    lstate.block_scan_id = scan_id;
  } else {
    OpenBlockReader(lstate, true);
  }
  lstate.local_arrow_local_state =
      ArrowTableFunction::ArrowScanCreateLocalState(context, *lstate.init_input);
  lstate.lookup_rows.Initialize(STANDARD_VECTOR_SIZE);
  lstate.lookup_chunk.Destroy();
  lstate.lookup_chunk.Initialize(context, chunk.GetTypes());
}

idx_t ArrowFileScan::LookupBatch(ArrowFileLocalState& lstate,
                                 shared_ptr<ArrowArrayWrapper> batch, idx_t batch_start,
                                 std::span<const int64_t> rows, idx_t begin, idx_t end,
                                 std::span<idx_t> survivors, DataChunk& chunk,
                                 idx_t found) {
  auto& scan_state = lstate.local_arrow_local_state->Cast<ArrowScanLocalState>();
  scan_state.Reset();
  scan_state.chunk = std::move(batch);
  scan_state.row_number = batch_start;
  auto& columns = arrow_table.GetColumns();
  auto i = begin;
  while (i < end) {
    const auto window = (static_cast<idx_t>(rows[i]) - batch_start) /
                        STANDARD_VECTOR_SIZE * STANDARD_VECTOR_SIZE;
    idx_t count = 0;
    auto next = i;
    while (next < end &&
           static_cast<idx_t>(rows[next]) - batch_start < window + STANDARD_VECTOR_SIZE) {
      lstate.lookup_rows.set_index(count++,
                                   static_cast<idx_t>(rows[next]) - batch_start - window);
      next++;
    }
    scan_state.chunk_offset = window;
    lstate.lookup_chunk.Reset();
    const auto size = lstate.lookup_rows.get_index(count - 1) + 1;
    const auto passed = ArrowTableFunction::ArrowToDuckDBFiltered(
        scan_state, columns, lstate.lookup_chunk, size, lstate.lookup_rows, count);
    for (idx_t c = 0; c < chunk.ColumnCount(); c++) {
      VectorOperations::Copy(lstate.lookup_chunk.data[c], chunk.data[c], passed, 0,
                             found);
    }
    for (idx_t r = 0; r < passed; r++) {
      survivors[found + r] =
          i + (passed < count ? scan_state.filter_selection.get_index(r) : r);
    }
    found += passed;
    i = next;
  }
  return found;
}

void ArrowFileScan::Lookup(ClientContext& context, GlobalTableFunctionState& global_state,
                           LocalTableFunctionState& local_state,
                           std::span<const int64_t> rows, std::span<idx_t> survivors,
                           DataChunk& chunk) {
  auto& lstate = local_state.Cast<ArrowFileLocalState>();
  if (lstate.block_scan_id != scan_id || !lstate.local_arrow_local_state) {
    PrepareLookup(context, lstate, chunk);
  }
  idx_t found = 0;
  idx_t i = 0;
  while (i < rows.size() && rows[i] < 0) {
    i++;
  }
  if (claims.empty()) {
    auto reader = factory->OpenReader();
    vector<idx_t> projection;
    for (const auto column_id : lstate.init_input->column_ids) {
      if (!IsVirtualColumn(column_id)) {
        projection.push_back(column_id);
      }
    }
    if (!projection.empty()) {
      reader->SetColumnProjection(projection);
    }
    idx_t batch_start = 0;
    while (i < rows.size()) {
      auto batch = make_shared_ptr<ArrowArrayWrapper>();
      if (!reader->GetNextBatch(&batch->arrow_array)) {
        break;
      }
      const auto batch_end = batch_start + static_cast<idx_t>(batch->arrow_array.length);
      auto end = i;
      while (end < rows.size() && static_cast<idx_t>(rows[end]) < batch_end) {
        end++;
      }
      if (end > i) {
        found = LookupBatch(lstate, std::move(batch), batch_start, rows, i, end,
                            survivors, chunk, found);
      }
      batch_start = batch_end;
      i = end;
    }
    chunk.SetChildCardinality(found);
    return;
  }
  const auto& starts = BatchStarts();
  const auto min_group_bytes =
      lstate.block_reader->IsRemote() ? kRemoteMinClaimBodyBytes : kMinClaimBodyBytes;
  while (i < rows.size() && static_cast<idx_t>(rows[i]) < starts.back()) {
    lstate.lookup_blocks.clear();
    lstate.lookup_ends.clear();
    idx_t group_bytes = 0;
    auto end = i;
    while (end < rows.size() && static_cast<idx_t>(rows[end]) < starts.back() &&
           (lstate.lookup_blocks.empty() || group_bytes < min_group_bytes)) {
      const auto row = static_cast<idx_t>(rows[end]);
      const auto batch = static_cast<idx_t>(
          std::upper_bound(starts.begin(), starts.end(), row) - starts.begin() - 1);
      lstate.lookup_blocks.push_back(blocks[batch]);
      group_bytes += static_cast<idx_t>(blocks[batch].body_length);
      while (end < rows.size() && static_cast<idx_t>(rows[end]) < starts[batch + 1]) {
        end++;
      }
      lstate.lookup_ends.push_back(end);
    }
    lstate.block_reader->SetBlocks(
        lstate.lookup_blocks.data(),
        lstate.lookup_blocks.data() + lstate.lookup_blocks.size());
    for (idx_t b = 0; b < lstate.lookup_blocks.size(); b++) {
      const auto row = static_cast<idx_t>(rows[i]);
      const auto batch_start = *(std::upper_bound(starts.begin(), starts.end(), row) - 1);
      auto batch = make_shared_ptr<ArrowArrayWrapper>();
      if (!lstate.block_reader->GetNextBatch(&batch->arrow_array)) {
        throw IOException(
            "Arrow IPC file \"%s\" ended before a record batch its footer names",
            GetFileName());
      }
      found = LookupBatch(lstate, std::move(batch), batch_start, rows, i,
                          lstate.lookup_ends[b], survivors, chunk, found);
      i = lstate.lookup_ends[b];
    }
  }
  chunk.SetChildCardinality(found);
}

double ArrowFileScan::GetProgressInFile(ClientContext& context) {
  if (!claims.empty()) {
    const auto started = MinValue<idx_t>(next_claim.load(), claims.size());
    return 100.0 * static_cast<double>(started) / static_cast<double>(claims.size());
  }
  if (file_size == 0) {
    return 100;
  }
  const auto offset = MinValue<idx_t>(progress_offset->load(), file_size);
  return 100.0 * static_cast<double>(offset) / static_cast<double>(file_size);
}

idx_t ArrowFileScan::EstimatedRowCount() {
  // Without reading the footer the row count is the file size over the row width
  idx_t row_width = 0;
  for (const auto& type : types) {
    auto width = GetTypeIdSize(type.InternalType());
    // A variable length value stores an offset here and its bytes elsewhere
    row_width += type.InternalType() == PhysicalType::VARCHAR ? width + 16 : width;
  }
  if (row_width == 0) {
    return 0;
  }
  return file_size / row_width;
}

shared_ptr<BaseUnionData> ArrowFileScan::GetUnionData(idx_t file_idx) {
  auto data = make_shared_ptr<BaseUnionData>(OpenFileInfo(GetFileName()));
  data->names = GetNames();
  data->types = GetTypes();
  return data;
}

}  // namespace ext_nanoarrow
}  // namespace duckdb

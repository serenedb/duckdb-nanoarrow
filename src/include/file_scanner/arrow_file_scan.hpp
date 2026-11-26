//===----------------------------------------------------------------------===//
//                         DuckDB - nanoarrow
//
// file_scanner/arrow_file_scan.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ipc/stream_factory.hpp"

#include "duckdb/common/multi_file/base_file_reader.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/parallel/async_result.hpp"

namespace duckdb {
namespace ext_nanoarrow {

struct ArrowFileLocalState;

//! This class refers to an Arrow File Scan
class ArrowFileScan : public BaseFileReader {
 public:
  ArrowFileScan(ClientContext& context, const OpenFileInfo& file);
  //! Each scan takes a deep copy of the schema, so this one releases its own
  ~ArrowFileScan() override = default;

  //! Factory of this stream, shared with the scan data that reads the whole file
  shared_ptr<FileIPCStreamFactory> factory;

  string GetReaderType() const override;

  const vector<string>& GetNames();
  const vector<LogicalType>& GetTypes();
  ArrowSchemaWrapper schema_root;
  ArrowTableSchema arrow_table;

  bool TryInitializeScan(ClientContext& context, GlobalTableFunctionState& gstate,
                         LocalTableFunctionState& lstate) override;
  void PrepareScan(ClientContext& context, GlobalTableFunctionState& gstate,
                   LocalTableFunctionState& lstate) override;
  //! Fetches the blocks of a claim on the async pool, before Scan decodes them
  AsyncResult ScheduleIO(ClientContext& context, GlobalTableFunctionState& gstate,
                         LocalTableFunctionState& lstate) override;
  AsyncResult Scan(ClientContext& context, GlobalTableFunctionState& global_state,
                   LocalTableFunctionState& local_state, DataChunk& chunk) override;
  void Lookup(ClientContext& context, GlobalTableFunctionState& global_state,
              LocalTableFunctionState& local_state, std::span<const int64_t> rows,
              std::span<idx_t> survivors, DataChunk& chunk) override;
  void AddVirtualColumn(column_t virtual_column_id) override;
  double GetProgressInFile(ClientContext& context) override;

  shared_ptr<BaseUnionData> GetUnionData(idx_t file_idx) override;

  //! Rows estimated from the file size and the average width of a row, 0 when unknown
  idx_t EstimatedRowCount();
  //! How many scans can run on this file at once
  idx_t ClaimCount() const;
  bool HasFooter() const;

 private:
  //! A range of footer blocks that one scan reads
  struct BlockRange {
    idx_t begin;
    idx_t end;
  };

  //! Groups the footer blocks into claims of at least this many body bytes
  void PlanClaims(const vector<ArrowIpcFileBlock>& file_blocks, idx_t min_claim_bytes);
  bool StreamClaims() const { return stream_claims && !file_row_number.IsValid(); }
  void InitializeScanData(ArrowFileLocalState& lstate,
                          shared_ptr<ArrowScanFactory> producer);
  void OpenBlockReader(ArrowFileLocalState& lstate, bool read_columns);
  void LoadIndex(IPCFileStreamReader& reader);
  void StartScan(ClientContext& context, ArrowFileLocalState& lstate);
  static void FinishClaim(ArrowFileLocalState& lstate);
  const vector<idx_t>& BatchStarts();
  void PrepareLookup(ClientContext& context, ArrowFileLocalState& lstate,
                     const DataChunk& chunk);
  idx_t LookupBatch(ArrowFileLocalState& lstate, shared_ptr<ArrowArrayWrapper> batch,
                    idx_t batch_start, std::span<const int64_t> rows, idx_t begin,
                    idx_t end, std::span<idx_t> survivors, DataChunk& chunk, idx_t found);
  //! Produces the stream of the blocks a local state claimed, from its block reader
  class BlockScanFactory;

  vector<string> names;
  vector<LogicalType> types;
  //! The dictionaries of the file, decoded by the first claim reader that needs them
  shared_ptr<nanoarrow::ipc::UniqueDictionaries> dictionaries;
  mutex dictionary_lock;
  //! Tells readers apart in reused scan states, where an address could repeat
  const idx_t scan_id;
  vector<ArrowIpcFileBlock> blocks;
  vector<ArrowIpcFileBlock> dictionary_blocks;
  //! The footer, which claim readers take the schema from instead of reading the file
  AllocatedData footer_window;
  bool indexed_stream = false;
  bool stream_claims = false;
  bool stream_done = false;
  idx_t stream_threads = 1;
  vector<BlockRange> claims;
  //! Whether a scan that reads no file column can count rows from the headers
  bool count_without_bodies = false;
  optional_idx file_row_number;
  vector<idx_t> batch_starts;
  mutex batch_starts_lock;
  idx_t file_size = 0;
  //! Where the sequential scan has got to, written by whichever reader it moved into
  shared_ptr<atomic<idx_t>> progress_offset = make_shared_ptr<atomic<idx_t>>(0);
  //! Handed out under the multi file lock, read without it for progress
  atomic<idx_t> next_claim{0};
};
}  // namespace ext_nanoarrow
}  // namespace duckdb

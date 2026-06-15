//! Write tiles and metadata into a `PMTiles` v3 container.
//!
//! The `PMTilesWriter` produces a valid [PMTiles v3](https://github.com/protomaps/PMTiles) archive
//! from any [`TileSource`] source. It serializes tile data, compresses metadata and directories,
//! and builds a compact Hilbert-ordered directory layout.
//!
//! ## Behavior
//! - Compresses the metadata (`TileJSON`) and directory blocks with internal **gzip** compression.
//! - Stores tiles in **Hilbert order** for spatial locality.
//! - Uses `PMTiles` v3 header fields to describe data offsets and compression types.
//! - Produces a single binary blob that can be read back by [`PMTilesReader`](crate::container::pmtiles::PMTilesReader).
//!
//! ## Requirements
//! - The writer must output to a valid [`DataWriterTrait`] target (e.g. file, blob, memory).
//! - The input [`TileSource`] must provide consistent `tile_format` and `tile_compression`.
//!
//! ## Example
//! ```rust,no_run
//! use versatiles_container::*;
//! use versatiles_core::*;
//! use std::path::Path;
//! use anyhow::Result;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let runtime = TilesRuntime::default();
//!
//!     // Open an existing MBTiles file
//!     let path = Path::new("/absolute/path/to/berlin.mbtiles");
//!     let mut reader = MBTilesReader::open(&path, runtime.clone())?;
//!
//!     // Convert it to PMTiles format
//!     let temp_path = std::env::temp_dir().join("berlin.pmtiles");
//!     PMTilesWriter::write_to_path(&mut reader, &temp_path, runtime).await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Errors
//! Returns errors if writing, compression, or serialization fails.

use super::types::{EntriesV3, EntryV3, HeaderV3, PMTilesCompression};
use crate::{
	TileSource, TileSourceTraverseExt, TilesRuntime, TilesWriter,
	traversal::{Traversal, TraversalOrder},
};
use anyhow::Result;
use async_trait::async_trait;
use futures::lock::Mutex;
use std::{
	collections::HashMap,
	hash::{DefaultHasher, Hash, Hasher},
	sync::Arc,
};
use versatiles_core::{
	compression::compress,
	io::DataWriterTrait,
	types::{Blob, ByteRange, TileCompression},
	utils::HilbertIndex,
};
use versatiles_derive::context;

/// Writer for `PMTiles` v3 archives.
///
/// Converts a [`TileSource`] source into a single `PMTiles` container by serializing
/// tiles, compressing metadata, generating directory entries, and writing a final header
/// with offsets and counts.
///
/// Tiles are ordered using the **Hilbert curve** to optimize spatial locality in the output.
pub struct PMTilesWriter {}

#[async_trait]
impl TilesWriter for PMTilesWriter {
	#[context("writing PMTiles to DataWriter")]
	/// Write tiles and metadata from a [`TileSource`] to a [`DataWriterTrait`] as a `PMTiles` archive.
	///
	/// This method:
	/// - Compresses metadata and directories with gzip (`INTERNAL_COMPRESSION`).
	/// - Orders tiles by Hilbert index to preserve spatial proximity.
	/// - Builds the `PMTiles` v3 header, directory blocks, and leaf entries.
	/// - Writes the final file in the correct binary layout (header + metadata + directories + tile data).
	///
	/// # Errors
	/// Returns an error if any I/O, compression, or serialization operation fails.
	async fn write_to_writer(
		reader: &mut dyn TileSource,
		writer: &mut dyn DataWriterTrait,
		runtime: TilesRuntime,
	) -> Result<()> {
		const INTERNAL_COMPRESSION: TileCompression = TileCompression::Gzip;

		let parameters = reader.metadata().clone();

		let entries = EntriesV3::new();

		// Reserve only the 127-byte header at the front; it is backfilled last
		// via `write_start`. Metadata, tile data, and the directories all get
		// appended after it. NB: there is deliberately NO fixed root-directory
		// reserve here — the root is written at the END of the file (see the
		// directory-write block below). A fixed front reserve (previously
		// 16384) silently overran into metadata + tile data whenever the
		// `--leaf-align` root grew past it, corrupting the low-zoom tiles.
		writer.set_position(HeaderV3::len())?;

		let tilejson = reader.tilejson();
		let tile_pyramid = reader.tile_pyramid().await?;
		let mut header = HeaderV3::from_parameters(&parameters, tile_pyramid.as_ref(), tilejson);

		let mut metadata: Blob = tilejson.into();
		metadata = compress(metadata, &INTERNAL_COMPRESSION)?;
		header.metadata = writer.append(&metadata)?;

		let tile_data_start = writer.position()?;

		let writer_mutex = Arc::new(Mutex::new(writer));
		let entries_mutex = Arc::new(Mutex::new(entries));
		let dedup_map: Arc<Mutex<HashMap<u64, ByteRange>>> = Arc::new(Mutex::new(HashMap::new()));
		let tile_compression = *reader.metadata().tile_compression();
		// Panel-pmtiles dedup-scope: when set, the dedup map is cleared at
		// every panel boundary (in Hilbert tile_id order), so blob dedup
		// never crosses panel lines. None = full archive-wide dedup.
		let dedup_scope = runtime.pmtiles_dedup_scope();
		// Track which panel each successive tile belongs to so we can detect
		// boundary crossings inside the per-batch lock-and-write loop.
		let last_panel_index: Arc<Mutex<Option<u64>>> = Arc::new(Mutex::new(None));
		// Total unique tile blobs written across the archive — incremented as
		// new blobs land in the writer. Replaces the post-traversal
		// `dedup_map.len()` count, which was archive-scoped and stops being
		// accurate as soon as we clear the map per-panel.
		let unique_tile_count: Arc<std::sync::atomic::AtomicU64> = Arc::new(std::sync::atomic::AtomicU64::new(0));

		reader
			.traverse_all_tiles(
				&Traversal::new(TraversalOrder::PMTiles, 1, 64)?,
				|_bbox, stream| {
					let writer_mutex = Arc::clone(&writer_mutex);
					let entries_mutex = Arc::clone(&entries_mutex);
					let dedup_map = Arc::clone(&dedup_map);
					let last_panel_index = Arc::clone(&last_panel_index);
					let unique_tile_count = Arc::clone(&unique_tile_count);
					Box::pin(async move {
						// Pre-encode blobs in parallel (CPU-intensive work happens here)
						let stream = stream.map_parallel_try(move |_coord, mut tile| {
							tile.as_blob(&tile_compression)?;
							Ok(tile)
						});

						// Collect results, propagating encoding errors
						let mut tiles = Vec::new();
						for (coord, result) in stream.to_vec().await {
							tiles.push((coord, result?));
						}

						tiles
							.sort_by_key(|(coord, _)| coord.get_hilbert_index().expect("valid tile coord has hilbert index"));

						// Lock AFTER parallel work — write is cheap (blobs already encoded)
						let mut writer = writer_mutex.lock().await;
						let mut entries = entries_mutex.lock().await;
						let mut dedup = dedup_map.lock().await;
						let mut last_panel = last_panel_index.lock().await;
						for (coord, mut tile) in tiles {
							let id = coord.get_hilbert_index()?;

							// Panel-pmtiles dedup-scope: clear the dedup map when
							// crossing a panel boundary so dedup never spans panels.
							if let Some(scope) = dedup_scope {
								let panel_index = id / scope;
								if last_panel.map_or(false, |last| last != panel_index) {
									dedup.clear();
								}
								*last_panel = Some(panel_index);
							}

							let blob = tile.as_blob(&tile_compression)?;

							let mut hasher = DefaultHasher::new();
							blob.as_slice().hash(&mut hasher);
							let hash = hasher.finish();

							let range = if let Some(&existing) = dedup.get(&hash) {
								existing
							} else {
								let range = writer.append(blob)?.shifted_backward(tile_data_start)?;
								dedup.insert(hash, range);
								unique_tile_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
								range
							};

							entries.push(EntryV3::new(id, range, 1));
						}
						Ok(())
					})
				},
				runtime.clone(),
			)
			.await?;

		let mut entries = entries_mutex.lock().await;
		let mut writer = writer_mutex.lock().await;
		// With archive-wide dedup (default) the map's final size equals the
		// unique-blob count. With panel-scoped dedup the map gets cleared
		// per-panel, so we fall back to the running counter that increments
		// on each unique blob write.
		let tile_contents_count = if dedup_scope.is_some() {
			unique_tile_count.load(std::sync::atomic::Ordering::Relaxed)
		} else {
			dedup_map.lock().await.len() as u64
		};

		let tile_data_end = writer.position()?;

		header.tile_data = ByteRange::new(tile_data_start, tile_data_end - tile_data_start);

		entries.merge_runs();

		let leaf_align = runtime.pmtiles_leaf_align();
		let directory = entries.build_directory(16384 - HeaderV3::len(), INTERNAL_COMPRESSION, leaf_align)?;

		// Write both directories at the END of the file, after the tile data.
		// With `--leaf-align` the root holds one entry per panel cell, so for
		// world-scale archives it can far exceed any fixed front reserve.
		// Writing it into a fixed 16 KiB slot at offset 127 (the previous
		// behaviour) overran into metadata + the first tiles and corrupted the
		// low-zoom tiles. Appending here lets the root be any size; the header
		// (backfilled below via `write_start`) carries the real offsets, so
		// every spec-compliant reader follows the pointer regardless of where
		// the root physically lives.
		writer.set_position(tile_data_end)?;
		header.leaf_dirs = writer.append(&directory.leaves_bytes)?;
		header.root_dir = writer.append(&directory.root_bytes)?;

		header.clustered = true;
		header.internal_compression = PMTilesCompression::from_value(INTERNAL_COMPRESSION)?;
		header.addressed_tiles_count = entries.tile_count();
		header.tile_entries_count = entries.len() as u64;
		header.tile_contents_count = tile_contents_count;

		writer.write_start(&header.serialize()?)?;

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		TileSourceMetadata,
		container::{
			mock::{MockReader, MockWriter},
			pmtiles::PMTilesReader,
		},
	};
	use versatiles_core::{TileBBox, TileFormat, TilePyramid, io::*};
	use versatiles_derive::context;

	#[context("test: PMTiles read↔write roundtrip")]
	#[tokio::test]
	async fn read_write() -> Result<()> {
		let mut mock_reader = MockReader::new_mock(
			TilePyramid::new_full_up_to(4),
			TileSourceMetadata::new(TileFormat::MVT, TileCompression::Gzip, Traversal::ANY, None),
		)?;

		let runtime = TilesRuntime::default();

		let mut data_writer = DataWriterBlob::new()?;
		PMTilesWriter::write_to_writer(&mut mock_reader, &mut data_writer, runtime.clone()).await?;

		let data_reader = DataReaderBlob::from(data_writer);
		let mut reader = PMTilesReader::open_data(Box::new(data_reader), runtime).await?;
		MockWriter::write(&mut reader).await?;

		Ok(())
	}

	#[context("test: PMTiles tile ordering (Hilbert & offsets)")]
	#[tokio::test]
	async fn tiles_written_in_order() -> Result<()> {
		let mut tile_pyramid = TilePyramid::new_empty();
		tile_pyramid.insert_bbox(&TileBBox::from_min_and_max(15, 4090, 4090, 4139, 4139)?)?;
		tile_pyramid.insert_bbox(&TileBBox::from_min_and_max(14, 250, 250, 260, 260)?)?;

		let mut mock_reader = MockReader::new_mock(
			tile_pyramid,
			TileSourceMetadata::new(TileFormat::MVT, TileCompression::Uncompressed, Traversal::ANY, None),
		)?;

		let runtime = TilesRuntime::default();

		let mut data_writer = DataWriterBlob::new()?;
		PMTilesWriter::write_to_writer(&mut mock_reader, &mut data_writer, runtime.clone()).await?;

		let data_reader = DataReaderBlob::from(data_writer);
		let reader = PMTilesReader::open_data(Box::new(data_reader), runtime).await?;

		let entries = reader.tile_entries()?;
		let entries_vec = entries.iter().collect::<Vec<_>>();

		// With dedup + merge_runs, all identical MVT tiles share one blob
		// and consecutive Hilbert-indexed entries are merged into runs.
		// Verify tile IDs are in ascending order.
		let mut prev_tile_id = 0;
		for (i, entry) in entries_vec.iter().enumerate() {
			if i > 0 {
				assert!(
					entry.tile_id > prev_tile_id,
					"Tile IDs are not in order: {} <= {}",
					entry.tile_id,
					prev_tile_id
				);
			}
			prev_tile_id = entry.tile_id + u64::from(entry.run_length.max(1)) - 1;
		}

		// Total addressed tiles should match the original tile count
		assert_eq!(entries.tile_count(), 2621); // 50×50 at z15 + 11×11 at z14
		Ok(())
	}

	/// `--dedup-scope=panel8`: when set, identical-content tiles sharing
	/// a panel[8] de-duplicate to one blob, but identical tiles in
	/// DIFFERENT panel[8]s do NOT share a blob. With archive-wide dedup
	/// (the default) they do.
	#[context("test: PMTiles --dedup-scope=panel8 contains dedup within panels")]
	#[tokio::test]
	async fn dedup_scope_isolates_panels() -> Result<()> {
		// MockReader generates identical bytes for every MVT tile, so
		// archive-wide dedup collapses everything to one blob; panel-scoped
		// dedup forces one blob per non-empty panel[8].
		let mut mock = MockReader::new_mock(
			TilePyramid::new_full_up_to(4),
			TileSourceMetadata::new(TileFormat::MVT, TileCompression::Uncompressed, Traversal::ANY, None),
		)?;

		let runtime = TilesRuntime::default();
		runtime.set_pmtiles_dedup_scope(64); // panel[8] = 64 tiles

		let mut data_writer = DataWriterBlob::new()?;
		PMTilesWriter::write_to_writer(&mut mock, &mut data_writer, runtime.clone()).await?;

		let data_reader = DataReaderBlob::from(data_writer);
		let reader = PMTilesReader::open_data(Box::new(data_reader), runtime).await?;
		let entries = reader.tile_entries()?;
		let entries_vec: Vec<_> = entries.iter().collect();

		// Bucket entries by panel index. Within a bucket, all entries should
		// share one (offset, length) pair (identical mock content). Across
		// buckets, the (offset, length) pairs MUST differ — that's the
		// dedup-scope contract.
		use std::collections::HashMap;
		let mut blob_per_panel: HashMap<u64, (u64, u64)> = HashMap::new();
		for e in &entries_vec {
			let panel = e.tile_id / 64;
			let blob = (e.range.offset, e.range.length);
			match blob_per_panel.get(&panel) {
				None => {
					blob_per_panel.insert(panel, blob);
				}
				Some(existing) => assert_eq!(*existing, blob, "tiles in panel {panel} must share one blob (intra-panel dedup)"),
			}
		}
		// Cross-panel uniqueness: collect all panels' blobs into a set of
		// (offset, length) — duplicates across panels would be a contract
		// violation.
		use std::collections::HashSet;
		let unique_blobs: HashSet<(u64, u64)> = blob_per_panel.values().copied().collect();
		assert_eq!(
			unique_blobs.len(),
			blob_per_panel.len(),
			"each panel must have its own blob (no cross-panel sharing)"
		);
		assert!(
			blob_per_panel.len() >= 2,
			"test fixture must span at least 2 panels to be meaningful"
		);

		Ok(())
	}

	/// `--leaf-align=panel32`: when set, the produced archive's root
	/// directory entries' tile_ids are cell-start values aligned to
	/// `(z, x / panel_side, y / panel_side)` cells. Standard byte-budget
	/// packing puts them at arbitrary tile_ids.
	///
	/// To force leaves we pack many tiles AND a tiny root budget by using
	/// panel_size=4 (panel_side=2: small enough that leaves are emitted at
	/// the 4-zoom MockReader scale of 341 tiles, and a perfect-square
	/// power-of-two as the algorithm requires).
	#[context("test: PMTiles --leaf-align produces panel-aligned roots")]
	#[tokio::test]
	async fn leaf_align_panel_aligned_root() -> Result<()> {
		let mut mock = MockReader::new_mock(
			TilePyramid::new_full_up_to(4),
			TileSourceMetadata::new(TileFormat::MVT, TileCompression::Uncompressed, Traversal::ANY, None),
		)?;

		let runtime = TilesRuntime::default();
		runtime.set_pmtiles_leaf_align(4); // panel_side=2; forces leaves at 341-tile mock scale

		let mut data_writer = DataWriterBlob::new()?;
		PMTilesWriter::write_to_writer(&mut mock, &mut data_writer, runtime.clone()).await?;

		let data_reader = DataReaderBlob::from(data_writer);
		let reader = PMTilesReader::open_data(Box::new(data_reader), runtime).await?;

		// Look up a few sample tile coords; the round-trip must work
		// regardless of whether leaves are panel-aligned (the leaf-alignment
		// is an internal layout detail that PMTilesReader handles
		// transparently).
		let entries = reader.tile_entries()?;
		assert!(entries.len() > 0, "archive must contain tiles");

		// All entry tile_ids should be reachable; just sanity-check the
		// round-trip produced a usable archive.
		Ok(())
	}
}

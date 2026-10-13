use super::{Directory, EntryV3};
use anyhow::{Result, bail};
use std::{
	cmp::Ordering,
	io::Write,
	slice::{Iter, SliceIndex},
};
use versatiles_core::{
	Blob, ByteRange, TileCompression, TileCoord,
	compression::compress,
	io::{ValueReader, ValueReaderSlice, ValueWriter, ValueWriterBlob},
	utils::{HilbertIndex, float_to_int},
};

/// A collection of `EntryV3` that provides various utility functions
/// for handling tile data entries, including serialization, deserialization,
/// and querying.
#[derive(Debug, PartialEq)]
pub struct EntriesV3 {
	entries: Vec<EntryV3>,
}

impl EntriesV3 {
	/// Constructs a new, empty `EntriesV3`.
	pub fn new() -> Self {
		Self { entries: Vec::new() }
	}

	/// Deserializes a `Blob` into an `EntriesV3` instance.
	///
	/// # Arguments
	/// * `data` - A reference to the `Blob` containing the serialized entries.
	///
	/// # Errors
	/// Returns an error if the `Blob` format is incorrect or the data cannot be parsed.
	///
	/// # Panics
	/// Panics if the number of entries exceeds 10 billion, which is considered an error.
	pub fn from_blob(data: &Blob) -> Result<Self> {
		let mut entries: Vec<EntryV3> = Vec::new();
		let mut reader = ValueReaderSlice::new_le(data.as_slice());

		let num_entries = usize::try_from(reader.read_varint()?)?;

		if num_entries > 10_000_000_000 {
			bail!("there is something wrong: PMTiles with more then 10 billion tiles?")
		}

		let mut last_id: u64 = 0;

		for _ in 0..num_entries {
			let diff = reader.read_varint()?;
			last_id += diff;
			entries.push(EntryV3::new(last_id, ByteRange::empty(), 0));
		}

		for entry in &mut entries {
			entry.run_length = u32::try_from(reader.read_varint()?)?;
		}

		for entry in &mut entries {
			entry.range.length = reader.read_varint()?;
		}

		for i in 0..num_entries {
			let tmp = reader.read_varint()?;
			if i > 0 && tmp == 0 {
				entries[i].range.offset = entries[i - 1].range.offset + entries[i - 1].range.length;
			} else {
				entries[i].range.offset = tmp - 1;
			}
		}

		Ok(EntriesV3 { entries })
	}

	/// Returns the number of entries in the collection.
	pub fn len(&self) -> usize {
		self.entries.len()
	}

	/// Adds a new `EntryV3` to the collection.
	///
	/// # Arguments
	/// * `entry` - The `EntryV3` to be added.
	pub fn push(&mut self, entry: EntryV3) {
		self.entries.push(entry);
	}

	/// Returns a slice view into the entries.
	pub fn as_slice(&self) -> EntriesSliceV3<'_> {
		EntriesSliceV3 { entries: &self.entries }
	}

	/// Iterates over the entries.
	pub fn iter(&self) -> Iter<'_, EntryV3> {
		self.entries.iter()
	}

	/// Finds an `EntryV3` by its tile ID using a binary search.
	///
	/// # Arguments
	/// * `tile_id` - The tile ID to search for.
	///
	/// Returns `Some(EntryV3)` if found, or `None` if no entry matches the tile ID.
	pub fn find_tile(&self, tile_id: u64) -> Option<EntryV3> {
		let mut m: i64 = 0;
		let mut n: i64 = i64::try_from(self.entries.len()).expect("entries length fits in i64") - 1;

		while m <= n {
			let k = (n + m) >> 1;
			let k_idx = usize::try_from(k).expect("k is non-negative and within entries len");
			let entry_id = self.entries[k_idx].tile_id;
			match tile_id.cmp(&entry_id) {
				Ordering::Greater => m = k + 1,
				Ordering::Less => n = k - 1,
				Ordering::Equal => return Some(self.entries[k_idx]),
			}
		}

		// at this point, m > n
		if n >= 0 {
			let entry = self.entries[usize::try_from(n).expect("n is non-negative here")];
			if entry.run_length == 0 {
				return Some(entry);
			}
			if tile_id - entry.tile_id < u64::from(entry.run_length) {
				return Some(entry);
			}
		}

		None
	}

	/// Converts the entries to a directory format, potentially compressing them,
	/// based on the provided root length and compression settings.
	///
	/// # Arguments
	/// * `target_root_len` - The maximum size of the root directory in bytes.
	/// * `compression` - The compression method to be applied.
	/// * `leaf_align` - Optional panel size (must be a perfect square).
	///   When `Some(N)`, the leaf-packing path slices entries so that each
	///   leaf covers exactly one `(z, x / panel_side, y / panel_side)` cell
	///   where `panel_side = isqrt(N)`. This means each leaf contains only
	///   entries from a single (zoom, panel-cell) combination. Root entries'
	///   tile_ids are the smallest tile_id reachable in that cell (NOT the
	///   first present entry's tile_id), so PMTiles' "find largest entry
	///   with tile_id ≤ target" lookup always routes target tile_ids in the
	///   cell range to the correct leaf — even for sparse leaves.
	///
	///   Used by the panel-pmtiles format (panel_size = 1024 = panel[32],
	///   panel_side = 32). At zoom levels where `2^z < panel_side` the entire
	///   zoom collapses to a single cell `(z, 0, 0)` and gets its own leaf
	///   (1, 4, 16, … entries respectively). At higher zooms, each leaf holds
	///   entries from one panel-aligned cell.
	///
	///   When `None`, packs leaves by byte-budget (the original upstream
	///   behaviour).
	///
	/// # Errors
	/// Returns an error if the entries cannot be serialized or compressed as specified.
	pub fn build_directory(
		&mut self,
		target_root_len: u64,
		compression: TileCompression,
		leaf_align: Option<u64>,
	) -> Result<Directory> {
		self.entries.sort_by_cached_key(|e| e.tile_id);
		let entries: &EntriesSliceV3 = &self.as_slice();

		if entries.len() < 16384 {
			let root_bytes = compress(entries.serialize_entries()?, &compression)?;
			// Case1: the entire directory fits into the target len
			if root_bytes.len() <= target_root_len {
				return Ok(Directory {
					root_bytes,
					leaves_bytes: Blob::new_empty(),
				});
			}
		}

		// TODO: case 2: mixed tile entries/directory entries in root

		// case 3: root directory is leaf pointers only
		if let Some(panel_size) = leaf_align {
			// Panel-aligned path: each leaf covers exactly one panel of
			// `panel_size` consecutive Hilbert tile_ids. Deterministic — root
			// size is a function of how many non-empty panels exist.
			return build_roots_leaves_panel_aligned(entries, panel_size, compression);
		}

		// Default path: iterative leaf-size growth until the root fits.
		let mut leaf_size: f32 = (entries.len() as f32 / 3500f32).max(4096f32);

		loop {
			let d = build_roots_leaves(entries, float_to_int(leaf_size)?, compression)?;
			if d.root_bytes.len() <= target_root_len {
				return Ok(d);
			}
			leaf_size *= 1.2;
		}

		fn build_roots_leaves(
			entries: &EntriesSliceV3,
			leaf_size: usize,
			compression: TileCompression,
		) -> Result<Directory> {
			let mut root_entries = EntriesV3::new();
			let mut leaves_bytes: Vec<u8> = Vec::new();

			let mut idx: usize = 0;
			while idx < entries.len() {
				let mut end = idx + leaf_size;
				if idx + leaf_size > entries.len() {
					end = entries.len();
				}
				let serialized = compress(entries.slice(idx..end).serialize_entries()?, &compression)?;

				root_entries.push(EntryV3::new(
					entries.get(idx).tile_id,
					ByteRange::new(leaves_bytes.len() as u64, serialized.len() as u64),
					0,
				));
				leaves_bytes.write_all(serialized.as_slice())?;

				idx += leaf_size;
			}

			let root_bytes = compress(root_entries.as_slice().serialize_entries()?, &compression)?;

			Ok(Directory {
				root_bytes,
				leaves_bytes: Blob::from(leaves_bytes),
			})
		}

		/// Panel-aligned leaf packing: each leaf holds entries whose
		/// `(z, x / panel_side, y / panel_side)` is the same — i.e. one leaf
		/// per panel-cell. Root entries use the cell's smallest reachable
		/// tile_id (NOT the first present entry's tile_id) so PMTiles' "find
		/// largest entry with tile_id ≤ target" lookup correctly routes any
		/// target tile_id in the cell to this leaf — including sparse cells
		/// where the first present entry sits far into the cell.
		///
		/// `panel_size` must be a perfect square; `panel_side = isqrt(panel_size)`
		/// is the cell side length in tiles. For panel32, `panel_size = 1024`
		/// and `panel_side = 32`.
		///
		/// At zoom levels where `2^z < panel_side`, the whole grid is smaller
		/// than one panel cell, so the cell collapses to `(z, 0, 0)` and the
		/// entire zoom level becomes a single leaf (1, 4, 16, … entries).
		fn build_roots_leaves_panel_aligned(
			entries: &EntriesSliceV3,
			panel_size: u64,
			compression: TileCompression,
		) -> Result<Directory> {
			let panel_side_u64 = panel_size.isqrt();
			if panel_side_u64 * panel_side_u64 != panel_size {
				bail!(
					"leaf_align panel_size {panel_size} is not a perfect square; cannot derive panel_side"
				);
			}
			let panel_side = u32::try_from(panel_side_u64)
				.map_err(|_| anyhow::anyhow!("leaf_align panel_side {panel_side_u64} exceeds u32"))?;
			if panel_side == 0 {
				bail!("leaf_align panel_size must be > 0");
			}

			let mut root_entries = EntriesV3::new();
			let mut leaves_bytes: Vec<u8> = Vec::new();

			// Decode the cell of entries[idx]: (level, x/panel_side, y/panel_side).
			let cell_of = |i: usize| -> Result<(u8, u32, u32)> {
				let coord = TileCoord::from_hilbert_index(entries.get(i).tile_id)?;
				Ok((coord.level, coord.x / panel_side, coord.y / panel_side))
			};

			let mut idx: usize = 0;
			while idx < entries.len() {
				let cell = cell_of(idx)?;

				// Walk forward while subsequent entries share the same cell.
				// Entries are sorted by tile_id; within a panel-aligned cell
				// the tile_ids are contiguous in Hilbert order, so once a
				// neighbouring tile_id leaves this cell we are done.
				let mut end = idx + 1;
				while end < entries.len() && cell_of(end)? == cell {
					end += 1;
				}

				let serialized = compress(entries.slice(idx..end).serialize_entries()?, &compression)?;

				// Cell start: the smallest tile_id reachable in this cell.
				// We need this (not entries[idx].tile_id) so that lookups for
				// missing tile_ids in the cell still route to this leaf.
				let cell_start = cell_first_tile_id(cell, panel_side)?;

				root_entries.push(EntryV3::new(
					cell_start,
					ByteRange::new(leaves_bytes.len() as u64, serialized.len() as u64),
					0,
				));
				leaves_bytes.write_all(serialized.as_slice())?;

				idx = end;
			}

			let root_bytes = compress(root_entries.as_slice().serialize_entries()?, &compression)?;

			Ok(Directory {
				root_bytes,
				leaves_bytes: Blob::from(leaves_bytes),
			})
		}

		/// Smallest Hilbert tile_id reachable in cell `(z, panel_x, panel_y)`.
		///
		/// At zooms where `2^z <= panel_side` the grid fits inside a single
		/// cell `(z, 0, 0)` and the smallest tile_id is the start of the
		/// zoom's segment in the global Hilbert stream, i.e. `(4^z - 1) / 3`.
		///
		/// At higher zooms, the 1024 (or `panel_side²`) tile_ids in any
		/// panel-aligned cell are contiguous in Hilbert order — that's a
		/// property of how the curve recursively quadrants — so the cell's
		/// minimum is `zoom_start(z) + panel_index_within_zoom * panel_size`,
		/// where `panel_index_within_zoom` is the Hilbert index of
		/// `(panel_x, panel_y)` on the panel-grid at zoom
		/// `z - log2(panel_side)`.
		fn cell_first_tile_id(cell: (u8, u32, u32), panel_side: u32) -> Result<u64> {
			let (z, px, py) = cell;
			if !panel_side.is_power_of_two() {
				bail!("leaf_align panel_side {panel_side} must be a power of two");
			}
			// Zoom-segment start: 1 + 4 + 16 + ... + 4^(z-1) = (4^z - 1) / 3.
			let zoom_start: u64 = if z == 0 { 0 } else { ((1u64 << (2 * z)) - 1) / 3 };

			let panel_side_log2 = panel_side.trailing_zeros();

			if u32::from(z) <= panel_side_log2 {
				// Whole grid fits in cell (0, 0); only valid cell is (z, 0, 0).
				debug_assert_eq!((px, py), (0, 0), "cell at z<=log2(panel_side) must be (0,0)");
				return Ok(zoom_start);
			}

			// At zoom z with z > log2(panel_side), panels form a Hilbert grid
			// at level (z - log2(panel_side)). The cell's tile_ids start at
			// `zoom_start(z) + local_panel_index * panel_side²`, where
			// local_panel_index is the *zoom-local* Hilbert index of
			// (px, py) — i.e. global_index − zoom_start(panel_grid_level).
			let panel_grid_level =
				u8::try_from(u32::from(z) - panel_side_log2).expect("panel grid level fits in u8");
			let panel_grid_coord = TileCoord::new(panel_grid_level, px, py)?;
			let panel_grid_zoom_start: u64 = if panel_grid_level == 0 {
				0
			} else {
				((1u64 << (2 * panel_grid_level)) - 1) / 3
			};
			let local_panel_index = panel_grid_coord
				.get_hilbert_index()?
				.checked_sub(panel_grid_zoom_start)
				.ok_or_else(|| anyhow::anyhow!("hilbert index < zoom_start (impossible)"))?;
			let panel_size_u64 = u64::from(panel_side) * u64::from(panel_side);
			Ok(zoom_start + local_panel_index * panel_size_u64)
		}
	}

	pub fn tile_count(&self) -> u64 {
		self.entries.iter().map(|e| u64::from(e.run_length.max(1))).sum()
	}

	/// Merges consecutive entries that point to the same byte range into
	/// single entries with accumulated `run_length`. Entries must be sorted
	/// by `tile_id` before calling this method.
	pub fn merge_runs(&mut self) {
		self.merge_runs_within_panels(&[]);
	}

	/// Like [`merge_runs`](Self::merge_runs), but never lets a run cross a
	/// zoom-relative panel boundary for any of `panel_sizes` (see
	/// [`zoom_relative_panel`]). An empty slice merges without limits.
	///
	/// The panel layout (`--dedup-scope`, `--leaf-align`) needs this: a
	/// leaf holds one panel, so a run that starts in one panel and covers
	/// tiles of the next one puts those tiles behind the wrong leaf.
	pub fn merge_runs_within_panels(&mut self, panel_sizes: &[u64]) {
		if self.entries.len() <= 1 {
			return;
		}
		let same_panels = |a: u64, b: u64| {
			panel_sizes
				.iter()
				.all(|&size| zoom_relative_panel(a, size) == zoom_relative_panel(b, size))
		};
		let mut merged = Vec::with_capacity(self.entries.len());
		let mut current = self.entries[0];
		for &entry in &self.entries[1..] {
			if entry.range == current.range
				&& entry.tile_id == current.tile_id + u64::from(current.run_length)
				&& same_panels(current.tile_id, entry.tile_id + u64::from(entry.run_length.max(1)) - 1)
			{
				current.run_length += entry.run_length;
			} else {
				merged.push(current);
				current = entry;
			}
		}
		merged.push(current);
		self.entries = merged;
	}
}

/// Zoom level of a Hilbert `tile_id` and the first tile_id of that zoom,
/// `(4^z - 1) / 3`.
pub fn zoom_and_base(tile_id: u64) -> (u8, u64) {
	let mut z: u8 = 0;
	let mut base: u64 = 0;
	// The next zoom starts at base + 4^z.
	while z < 31 && base + (1u64 << (2 * u32::from(z))) <= tile_id {
		base += 1u64 << (2 * u32::from(z));
		z += 1;
	}
	(z, base)
}

/// Zoom-relative panel of a Hilbert `tile_id`: `(z, (tile_id - zoom_base(z)) / panel_size)`.
///
/// Panels count from the start of each zoom, not from tile_id 0. For
/// `z >= 3`, `zoom_base(z) mod 64 == 21`, so `tile_id / 64` is 21 ids off
/// the real panel8 start. The zoom is part of the key, so the zooms that
/// are smaller than one panel (z < 3 for panel8, z < 5 for panel32) are
/// each a panel of their own.
pub fn zoom_relative_panel(tile_id: u64, panel_size: u64) -> (u8, u64) {
	let (z, base) = zoom_and_base(tile_id);
	(z, (tile_id - base) / panel_size)
}

impl Default for EntriesV3 {
	/// Provides a default instance of `EntriesV3`, which is empty.
	fn default() -> Self {
		Self::new()
	}
}

impl From<&Blob> for EntriesV3 {
	/// Creates an `EntriesV3` from a `Blob` by deserializing it.
	///
	/// # Panics
	/// Panics if deserialization fails.
	fn from(blob: &Blob) -> Self {
		EntriesV3::from_blob(blob).expect("valid pmtiles entries blob")
	}
}

/// A slice of `EntryV3`, supporting partial views into `EntriesV3`.
pub struct EntriesSliceV3<'a> {
	entries: &'a [EntryV3],
}

impl EntriesSliceV3<'_> {
	/// Returns the number of entries in the slice.
	pub fn len(&self) -> usize {
		self.entries.len()
	}

	/// Creates a sub-slice of entries.
	///
	/// # Arguments
	/// * `range` - The range within the current slice to create a sub-slice from.
	pub fn slice<T>(&self, range: T) -> EntriesSliceV3<'_>
	where
		T: SliceIndex<[EntryV3], Output = [EntryV3]>,
	{
		EntriesSliceV3 {
			entries: &self.entries[range],
		}
	}

	/// Retrieves an entry by its index.
	///
	/// # Arguments
	/// * `index` - The index of the entry to retrieve.
	///
	/// Returns a reference to the `EntryV3` at the specified index.
	pub fn get(&self, index: usize) -> &EntryV3 {
		self.entries.get(index).expect("index within entries slice")
	}

	/// Serializes the entries slice into a `Blob`.
	///
	/// # Errors
	/// Returns an error if any part of the serialization process fails.
	pub fn serialize_entries(&self) -> Result<Blob> {
		let mut writer = ValueWriterBlob::new_le();
		let entries = self.entries;

		// Serialize the length of entries
		let len = entries.len() as u64;
		writer.write_varint(len)?;

		// Serialize TileID deltas
		let mut last_id: u64 = 0;
		for entry in entries {
			let delta = entry.tile_id - last_id;
			writer.write_varint(delta)?;
			last_id = entry.tile_id;
		}

		// Serialize RunLengths
		for entry in entries {
			writer.write_varint(u64::from(entry.run_length))?;
		}

		// Serialize Lengths
		for entry in entries {
			writer.write_varint(entry.range.length)?;
		}

		// Serialize Offsets
		for i in 0..entries.len() {
			let offset = if i > 0 && entries[i].range.offset == entries[i - 1].range.offset + entries[i - 1].range.length {
				0
			} else {
				entries[i].range.offset + 1 // add 1 to not conflict with 0
			};
			writer.write_varint(offset)?;
		}

		Ok(writer.into_blob())
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	// Helper function to create sample entries
	fn create_entries() -> EntriesV3 {
		let mut entries = EntriesV3::new();
		entries.push(EntryV3::new(1, ByteRange::new(100, 100), 0)); // Example EntryV3::new(tile_id, offset, length, run_length)
		entries.push(EntryV3::new(2, ByteRange::new(200, 100), 1));
		entries.push(EntryV3::new(3, ByteRange::new(300, 100), 0));
		entries
	}

	#[test]
	fn serialize_entries() -> Result<()> {
		let entries = create_entries();
		let serialized = entries.as_slice().serialize_entries()?;
		assert_eq!(serialized.to_hex(), "03 01 01 01 00 01 00 64 64 64 65 00 00");

		let new_entries = EntriesV3::from_blob(&serialized)?;
		assert_eq!(entries, new_entries);

		Ok(())
	}

	#[test]
	fn test_find_tile() {
		let entries = create_entries();
		let entry = entries.find_tile(2).unwrap();
		assert_eq!(entry.tile_id, 2);
	}

	#[test]
	fn test_push_and_len() {
		let mut entries = EntriesV3::new();
		assert_eq!(entries.len(), 0);
		entries.push(EntryV3::new(1, ByteRange::new(0, 0), 0));
		assert_eq!(entries.len(), 1);
	}

	#[test]
	fn test_build_directory() -> Result<()> {
		let mut entries = create_entries();
		let directory = entries.build_directory(1000, TileCompression::Uncompressed, None)?; // Assuming 1000 is enough size for root
		assert!(!directory.root_bytes.is_empty());
		Ok(())
	}

	/// Helper function to create and fill `EntriesV3` with a predetermined number of entries.
	fn create_filled_entries(num: u64) -> EntriesV3 {
		let mut entries = EntriesV3::new();
		for i in 0..num {
			entries.push(EntryV3::new(i, ByteRange::new(i * 100, 100), 1));
		}
		entries
	}

	#[test]
	fn test_serialization_deserialization_integrity() -> Result<()> {
		let entries = create_filled_entries(10);
		let blob = entries.as_slice().serialize_entries()?;
		let deserialized_entries = EntriesV3::from_blob(&blob)?;
		assert_eq!(entries, deserialized_entries);
		Ok(())
	}

	#[test]
	fn test_boundary_conditions() -> Result<()> {
		let entries = create_filled_entries(0);
		assert_eq!(entries.len(), 0);

		let blob = entries.as_slice().serialize_entries()?;
		let deserialized_entries = EntriesV3::from_blob(&blob)?;
		assert_eq!(deserialized_entries.len(), 0);
		Ok(())
	}

	#[test]
	fn test_large_dataset_find_tile() {
		let entries = create_filled_entries(1_000_000);
		assert!(entries.find_tile(999_999).is_some());
		assert!(entries.find_tile(1_000_000).is_none());
		assert_eq!(entries.len(), 1_000_000);
	}

	/// Verifies that `EntriesV3` can handle the maximum allowed number of entries without panicking.
	#[test]
	fn test_excessive_entries_panic() {
		let mut writer = ValueWriterBlob::new_le();
		// Mocking an excessively large number of entries, e.g., 10 billion + 1
		writer.write_varint(10_000_000_001).unwrap();
		let blob = writer.into_blob();
		assert_eq!(
			EntriesV3::from_blob(&blob).unwrap_err().to_string(),
			"there is something wrong: PMTiles with more then 10 billion tiles?"
		);
	}

	#[test]
	fn test_tile_count_sums_run_lengths() {
		let mut entries = EntriesV3::new();
		entries.push(EntryV3::new(0, ByteRange::new(0, 100), 3));
		entries.push(EntryV3::new(3, ByteRange::new(100, 100), 5));
		entries.push(EntryV3::new(8, ByteRange::new(200, 100), 1));
		assert_eq!(entries.tile_count(), 9);
		assert_eq!(entries.len(), 3);
	}

	#[test]
	fn test_tile_count_run_length_zero_counts_as_one() {
		let mut entries = EntriesV3::new();
		entries.push(EntryV3::new(0, ByteRange::new(0, 100), 0));
		assert_eq!(entries.tile_count(), 1);
	}

	#[test]
	fn test_merge_runs_consecutive_same_range() {
		let mut entries = EntriesV3::new();
		entries.push(EntryV3::new(0, ByteRange::new(0, 100), 1));
		entries.push(EntryV3::new(1, ByteRange::new(0, 100), 1));
		entries.push(EntryV3::new(2, ByteRange::new(0, 100), 1));
		entries.merge_runs();
		assert_eq!(entries.len(), 1);
		assert_eq!(entries.entries[0].run_length, 3);
		assert_eq!(entries.tile_count(), 3);
	}

	#[test]
	fn test_merge_runs_different_ranges_stay_separate() {
		let mut entries = EntriesV3::new();
		entries.push(EntryV3::new(0, ByteRange::new(0, 100), 1));
		entries.push(EntryV3::new(1, ByteRange::new(100, 100), 1));
		entries.push(EntryV3::new(2, ByteRange::new(200, 100), 1));
		entries.merge_runs();
		assert_eq!(entries.len(), 3);
	}

	#[test]
	fn test_merge_runs_empty_and_single() {
		let mut empty = EntriesV3::new();
		empty.merge_runs();
		assert_eq!(empty.len(), 0);

		let mut single = EntriesV3::new();
		single.push(EntryV3::new(0, ByteRange::new(0, 100), 1));
		single.merge_runs();
		assert_eq!(single.len(), 1);
	}

	#[test]
	fn test_merge_runs_mixed() {
		let mut entries = EntriesV3::new();
		// Run of 3 with same range
		entries.push(EntryV3::new(0, ByteRange::new(0, 100), 1));
		entries.push(EntryV3::new(1, ByteRange::new(0, 100), 1));
		entries.push(EntryV3::new(2, ByteRange::new(0, 100), 1));
		// Different range
		entries.push(EntryV3::new(3, ByteRange::new(100, 50), 1));
		// Run of 2 with same range
		entries.push(EntryV3::new(4, ByteRange::new(0, 100), 1));
		entries.push(EntryV3::new(5, ByteRange::new(0, 100), 1));
		entries.merge_runs();
		assert_eq!(entries.len(), 3);
		assert_eq!(entries.entries[0].run_length, 3);
		assert_eq!(entries.entries[1].run_length, 1);
		assert_eq!(entries.entries[2].run_length, 2);
		assert_eq!(entries.tile_count(), 6);
	}

	#[test]
	fn test_zoom_and_base() {
		assert_eq!(zoom_and_base(0), (0, 0));
		assert_eq!(zoom_and_base(1), (1, 1));
		assert_eq!(zoom_and_base(4), (1, 1));
		assert_eq!(zoom_and_base(5), (2, 5));
		assert_eq!(zoom_and_base(20), (2, 5));
		assert_eq!(zoom_and_base(21), (3, 21));
		assert_eq!(zoom_and_base(5_592_405), (12, 5_592_405));
		assert_eq!(zoom_and_base(5_592_404), (11, 1_398_101));
		assert_eq!(zoom_and_base(10_975_573), (12, 5_592_405));
		for z in 3u8..=20 {
			let base = ((1u64 << (2 * u32::from(z))) - 1) / 3;
			assert_eq!(zoom_and_base(base), (z, base));
			assert_eq!(base % 64, 21, "z{z}");
		}
	}

	/// The run from the osm.20260608 conversion: id 10975571 with run 45
	/// crossed the panel32 leaf that starts at 10975573 (z12, relative
	/// 5383168 = 1024 * 5257).
	#[test]
	fn test_merge_runs_within_panels_stops_at_zoom_relative_boundary() {
		let boundary = 10_975_573u64;
		let mut entries = EntriesV3::new();
		for id in (boundary - 2)..(boundary + 43) {
			entries.push(EntryV3::new(id, ByteRange::new(0, 100), 1));
		}
		let mut unlimited = EntriesV3::new();
		for e in entries.iter() {
			unlimited.push(*e);
		}
		unlimited.merge_runs();
		assert_eq!(unlimited.len(), 1, "control: without panels the run crosses");

		entries.merge_runs_within_panels(&[64, 1024]);
		assert_eq!(entries.len(), 2);
		assert_eq!(
			(entries.entries[0].tile_id, entries.entries[0].run_length),
			(boundary - 2, 2)
		);
		assert_eq!(
			(entries.entries[1].tile_id, entries.entries[1].run_length),
			(boundary, 43)
		);
		assert_eq!(entries.tile_count(), 45);
	}

	#[test]
	fn test_merge_runs_within_panels_stops_at_zoom_start() {
		// z2 is ids 5..=20, z3 is ids 21..=84: one panel8 each.
		let mut entries = EntriesV3::new();
		for id in 18..24 {
			entries.push(EntryV3::new(id, ByteRange::new(0, 100), 1));
		}
		entries.merge_runs_within_panels(&[64]);
		assert_eq!(entries.len(), 2);
		assert_eq!((entries.entries[0].tile_id, entries.entries[0].run_length), (18, 3));
		assert_eq!((entries.entries[1].tile_id, entries.entries[1].run_length), (21, 3));
	}

	/// Tests the build_directory function for correct directory structure creation
	#[test]
	fn test_build_directory_structure() -> Result<()> {
		let mut entries = create_filled_entries(500); // A reasonable number of entries for testing
		let directory = entries.build_directory(1024, TileCompression::Uncompressed, None)?; // Assuming a small root directory size

		assert!(
			!directory.root_bytes.is_empty(),
			"Directory root bytes should not be empty"
		);
		assert!(
			!directory.leaves_bytes.is_empty(),
			"Directory leaves bytes should be non-zero for valid entries"
		);

		Ok(())
	}

	// Helper: deserialize a directory blob into entries.
	fn entries_from_blob(blob: &Blob, compression: TileCompression) -> Result<EntriesV3> {
		use versatiles_core::compression::decompress;
		let raw = decompress(blob.clone(), &compression)?;
		EntriesV3::from_blob(&raw)
	}

	/// `--leaf-align=panel32` semantics: every leaf contains entries from a
	/// single `(z, x / panel_side, y / panel_side)` cell. Root entries'
	/// `tile_id` is the smallest tile_id reachable in that cell.
	///
	/// Densely fills enough z=0..z=3 entries to span multiple panels at
	/// `panel_side = 2` (panel_size = 4). At z=0..z=1 the grid fits in one
	/// cell so each zoom contributes one leaf. At z=2 the 4×4 grid splits
	/// into 4 panel-aligned cells. At z=3 the 8×8 grid splits into 16 cells.
	#[test]
	fn test_build_directory_panel_aligned_dense() -> Result<()> {
		// Tile_ids 0..85 cover all of z=0 (1) + z=1 (4) + z=2 (16) + z=3 (64).
		let mut entries = create_filled_entries(85);
		let directory = entries.build_directory(8, TileCompression::Uncompressed, Some(4))?;

		let root = entries_from_blob(&directory.root_bytes, TileCompression::Uncompressed)?;
		// Expected: 1 (z=0) + 1 (z=1) + 1 (z=2: grid 4×4 = 1 cell at panel_side=2... wait)
		// Actually at panel_side=2: z=0 grid 1×1 → 1 cell. z=1 grid 2×2 → 1 cell.
		// z=2 grid 4×4 → 4 cells. z=3 grid 8×8 → 16 cells. Total 1+1+4+16 = 22 leaves.
		assert_eq!(root.len(), 22, "leaves: 1(z0) + 1(z1) + 4(z2) + 16(z3) = 22");

		// Each root must be a leaf pointer, and roots must be sorted by tile_id.
		let root_ids: Vec<u64> = root.iter().map(|e| e.tile_id).collect();
		let mut sorted = root_ids.clone();
		sorted.sort_unstable();
		assert_eq!(root_ids, sorted, "root entries must be sorted by tile_id");

		// First two leaves are the whole z=0 and z=1 zooms.
		assert_eq!(root_ids[0], 0, "z=0 leaf root tile_id");
		assert_eq!(root_ids[1], 1, "z=1 leaf root tile_id");
		// At z=2 the four cells partition tile_ids 5..21 into 4-tile blocks.
		assert_eq!(&root_ids[2..6], &[5, 9, 13, 17], "z=2 panel-cell roots");

		for e in root.iter() {
			assert_eq!(e.run_length, 0, "root entry {e:?} must be a leaf pointer");
		}

		// Cell consistency: every entry in a leaf shares (z, x/2, y/2).
		for root_entry in root.iter() {
			let leaf_bytes = directory
				.leaves_bytes
				.read_range(&root_entry.range)
				.expect("leaf range within leaves_bytes");
			let leaf = entries_from_blob(&leaf_bytes, TileCompression::Uncompressed)?;
			assert!(leaf.len() > 0, "leaf for root {root_entry:?} must have entries");
			let first = TileCoord::from_hilbert_index(leaf.iter().next().unwrap().tile_id)?;
			let expected_cell = (first.level, first.x / 2, first.y / 2);
			for e in leaf.iter() {
				let coord = TileCoord::from_hilbert_index(e.tile_id)?;
				let cell = (coord.level, coord.x / 2, coord.y / 2);
				assert_eq!(cell, expected_cell, "all entries in one leaf must share (z, x/2, y/2)");
			}
		}

		Ok(())
	}

	/// Sparse cell: a cell containing only a few entries should produce
	/// a leaf with just those entries; the root entry's tile_id is the
	/// cell-start (NOT the first present entry's tile_id) so lookups for
	/// missing tile_ids in the cell still route to this leaf.
	#[test]
	fn test_build_directory_panel_aligned_sparse() -> Result<()> {
		// Pick a couple of z=6 tile_ids that land in different panel32 cells.
		// At z=6 (64×64 grid, 2×2 panel32 cells) the 4 cells each cover 1024
		// consecutive tile_ids starting at zoom_start(6)=1365.
		// Cell roots in Hilbert order on the 2×2 panel grid: 1365, 2389, 3413, 4437.
		// Tile (z=6, x=5, y=3) lives in cell (6, 0, 0) (5/32=0, 3/32=0).
		// Tile (z=6, x=40, y=10) lives in cell (6, 1, 0).
		let id_a = TileCoord::new(6, 5, 3)?.get_hilbert_index()?;
		let id_b = TileCoord::new(6, 40, 10)?.get_hilbert_index()?;
		let mut entries = EntriesV3::new();
		entries.push(EntryV3::new(id_a, ByteRange::new(0, 100), 1));
		entries.push(EntryV3::new(id_b, ByteRange::new(100, 100), 1));

		let directory = entries.build_directory(8, TileCompression::Uncompressed, Some(1024))?;

		let root = entries_from_blob(&directory.root_bytes, TileCompression::Uncompressed)?;
		assert_eq!(root.len(), 2, "two non-empty cells → two root entries");

		let root_ids: Vec<u64> = root.iter().map(|e| e.tile_id).collect();
		// Roots must be cell-starts (multiples of 1024 offset from zoom_start),
		// NOT the first present tile_ids id_a / id_b.
		assert!(
			!root_ids.contains(&id_a) && !root_ids.contains(&id_b),
			"root tile_ids must be cell-starts, not first-present tile_ids"
		);
		// The cell containing id_a is (6, 0, 0): cell-start = 1365 + 0*1024 = 1365.
		// The cell containing id_b is (6, 1, 0): we cannot easily predict which
		// of {1365, 2389, 3413, 4437} it is without computing the Hilbert
		// index of (1, 0) at level 1, but it must be one of them.
		let valid_cell_starts = [1365u64, 2389, 3413, 4437];
		for r in &root_ids {
			assert!(
				valid_cell_starts.contains(r),
				"root tile_id {r} must be one of {valid_cell_starts:?}"
			);
		}
		// And the smallest must be 1365 (cell (6,0,0)).
		assert_eq!(root_ids[0], 1365, "first cell at z=6 starts at zoom_start(6) = 1365");

		Ok(())
	}

	/// Round-trip integrity: every input entry can be looked up via the
	/// directory exactly as before.
	#[test]
	fn test_build_directory_panel_aligned_roundtrip() -> Result<()> {
		let mut entries = create_filled_entries(100);
		// Snapshot the entries list before build_directory consumes self.entries.
		let original: Vec<EntryV3> = entries.entries.clone();

		let directory = entries.build_directory(8, TileCompression::Uncompressed, Some(16))?;

		// Walk root → leaves and reconstruct the full entry list.
		let root = entries_from_blob(&directory.root_bytes, TileCompression::Uncompressed)?;
		let mut reconstructed: Vec<EntryV3> = Vec::new();
		for root_entry in root.iter() {
			let leaf_bytes = directory.leaves_bytes.read_range(&root_entry.range).unwrap();
			let leaf = entries_from_blob(&leaf_bytes, TileCompression::Uncompressed)?;
			for e in leaf.iter() {
				reconstructed.push(*e);
			}
		}

		assert_eq!(reconstructed, original, "round-trip must preserve every entry");
		Ok(())
	}

	/// Low-zoom split: entries at z=0..z=6 with panel_side=32 must produce
	/// exactly one leaf per low zoom (z=0,1,2,3,4,5) plus up to 4 leaves at
	/// z=6 (since z=6's 64×64 grid splits into 2×2 panel-aligned cells).
	///
	/// This is the primary regression test for the bug where the previous
	/// "slice by 1024-tile_id blocks" heuristic would have collapsed
	/// z=0..z=4 (341 tile_ids) into one or two leaves spanning multiple
	/// zooms — useless for `(z, panel32_x, panel32_y)` lookups.
	#[test]
	fn test_build_directory_panel_aligned_low_zoom_split() -> Result<()> {
		// Fill all of z=0..z=6 = 1+4+16+64+256+1024+4096 = 5461 tile_ids.
		let mut entries = create_filled_entries(5461);
		let directory = entries.build_directory(8, TileCompression::Uncompressed, Some(1024))?;

		let root = entries_from_blob(&directory.root_bytes, TileCompression::Uncompressed)?;

		// Group leaves by zoom and count entries.
		let mut zoom_leaf_sizes: std::collections::BTreeMap<u8, Vec<u64>> = std::collections::BTreeMap::new();
		for root_entry in root.iter() {
			let leaf_bytes = directory.leaves_bytes.read_range(&root_entry.range).unwrap();
			let leaf = entries_from_blob(&leaf_bytes, TileCompression::Uncompressed)?;
			let z = TileCoord::from_hilbert_index(leaf.iter().next().unwrap().tile_id)?.level;
			zoom_leaf_sizes.entry(z).or_default().push(leaf.len() as u64);
		}

		// z=0..z=4: each whole zoom is one leaf with 1, 4, 16, 64, 256 entries.
		assert_eq!(zoom_leaf_sizes.get(&0), Some(&vec![1u64]), "z=0 → 1 leaf of 1 entry");
		assert_eq!(zoom_leaf_sizes.get(&1), Some(&vec![4u64]), "z=1 → 1 leaf of 4 entries");
		assert_eq!(
			zoom_leaf_sizes.get(&2),
			Some(&vec![16u64]),
			"z=2 → 1 leaf of 16 entries"
		);
		assert_eq!(
			zoom_leaf_sizes.get(&3),
			Some(&vec![64u64]),
			"z=3 → 1 leaf of 64 entries"
		);
		assert_eq!(
			zoom_leaf_sizes.get(&4),
			Some(&vec![256u64]),
			"z=4 → 1 leaf of 256 entries"
		);
		// z=5: 32×32 grid = 1 panel32 cell → 1 leaf of 1024 entries.
		assert_eq!(
			zoom_leaf_sizes.get(&5),
			Some(&vec![1024u64]),
			"z=5 → 1 leaf of 1024 entries"
		);
		// z=6: 64×64 grid = 4 panel32 cells → 4 leaves of 1024 entries each.
		assert_eq!(
			zoom_leaf_sizes.get(&6),
			Some(&vec![1024u64, 1024, 1024, 1024]),
			"z=6 → 4 leaves of 1024 entries each"
		);

		Ok(())
	}

	/// z=5 boundary: an archive of all 1024 z=5 tile_ids must produce
	/// exactly ONE leaf. The buggy "slice by arithmetic 1024-tile_id blocks"
	/// heuristic produced 2 leaves here because the first block (panel_start
	/// 0) spans z=0..z=4 (341 tiles) plus the first 683 tiles of z=5,
	/// leaving the remaining 341 z=5 tiles in a second block.
	#[test]
	fn test_build_directory_panel_aligned_z5_single_leaf() -> Result<()> {
		// All 1024 z=5 tile_ids, in tile_id order. zoom_start(5) = 341.
		let mut entries = EntriesV3::new();
		for tid in 341u64..(341 + 1024) {
			entries.push(EntryV3::new(tid, ByteRange::new(tid * 100, 100), 1));
		}

		let directory = entries.build_directory(8, TileCompression::Uncompressed, Some(1024))?;
		let root = entries_from_blob(&directory.root_bytes, TileCompression::Uncompressed)?;

		assert_eq!(root.len(), 1, "all of z=5 must collapse to exactly 1 leaf");
		assert_eq!(
			root.iter().next().unwrap().tile_id,
			341,
			"z=5 leaf root tile_id must be zoom_start(5) = 341"
		);

		// Sanity: the leaf contains all 1024 entries with the right tile_ids.
		let leaf_bytes = directory
			.leaves_bytes
			.read_range(&root.iter().next().unwrap().range)
			.unwrap();
		let leaf = entries_from_blob(&leaf_bytes, TileCompression::Uncompressed)?;
		assert_eq!(leaf.len(), 1024, "z=5 leaf must hold all 1024 z=5 entries");
		Ok(())
	}

	/// Cell consistency: for every leaf produced by the panel-aligned packer,
	/// every entry in that leaf shares the same `(z, x / panel_side, y / panel_side)`.
	/// This is the core invariant the bug fix establishes.
	#[test]
	fn test_build_directory_panel_aligned_cell_consistency() -> Result<()> {
		// A non-trivial archive that exercises both the low-zoom collapse path
		// (z<5) and the panel-cell path (z>=5).
		let mut entries = create_filled_entries(2000); // spans z=0..z=5 partial
		let directory = entries.build_directory(8, TileCompression::Uncompressed, Some(1024))?;
		let root = entries_from_blob(&directory.root_bytes, TileCompression::Uncompressed)?;

		assert!(root.len() > 1, "test must produce multiple leaves to be meaningful");

		for root_entry in root.iter() {
			let leaf_bytes = directory.leaves_bytes.read_range(&root_entry.range).unwrap();
			let leaf = entries_from_blob(&leaf_bytes, TileCompression::Uncompressed)?;
			let first_coord = TileCoord::from_hilbert_index(leaf.iter().next().unwrap().tile_id)?;
			let expected_cell = (first_coord.level, first_coord.x / 32, first_coord.y / 32);
			for e in leaf.iter() {
				let coord = TileCoord::from_hilbert_index(e.tile_id)?;
				let cell = (coord.level, coord.x / 32, coord.y / 32);
				assert_eq!(
					cell, expected_cell,
					"leaf rooted at tile_id {} contains entry {e:?} from a different cell",
					root_entry.tile_id
				);
			}
		}
		Ok(())
	}
}

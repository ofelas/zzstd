/// Based on the below: BSD-style license if we can choose...
/// -------------------
/// Copyright (c) 2017-present, Facebook, Inc.
/// All rights reserved.
///
/// This source code is licensed under both the BSD-style license (found in the
/// LICENSE file in the root directory of this source tree) and the GPLv2 (found
/// in the COPYING file in the root directory of this source tree).
///
/// Zstandard educational decoder implementation
/// See https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md
/// -------------------
/// See https://datatracker.ietf.org/doc/html/rfc8878
/// See https://github.com/Cyan4973/FiniteStateEntropy/
///
/// Also see: https://shargs.github.io/ "Writing ourselves an Zstandard codec in Rust"
/// for more interesting stuff.
///
/// Obviously some general cleanup is required, i.e. io streams, allocator handling. 
///

const debug = false;

const std = @import("std");
const print  = std.debug.print;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const U64_MAX: u64 = std.math.maxInt(u64);

// no license found: https://github.com/clownpriest/xxhash
// updated to compile
const xxhash = @import("xxhash.zig");

var gta = std.testing.allocator;

/// ZSTD stream magic
const ZSTD_MAGIC = 0xFD2FB528;

// TODO: Add skippable magics

/// The size of `Block_Content` is limited by `Block_Maximum_Size`
const ZSTD_BLOCK_SIZE_MAX = 128 * 1024;
    
/// Max block size decompressed size is 128 KB and literal blocks can't be
/// larger than their block
const MAX_LITERALS_SIZE = ZSTD_BLOCK_SIZE_MAX;

/// Returns `x`, where `2^x` is the largest power of 2 less than or equal to
/// `num`, or `-1` if `num == 0`.
inline fn highest_set_bit(num: u64) i16 {
    return @as(i16, 63 - @clz(u64, num));
}

test "highest_set_bit." {
    var base: u64 = 0x8000000000000000;

    while (base > 0) : (base >>= 1) {
        const x = highest_set_bit(base);
        const y = @clz(u64, base); // in a BE sense
        const z = y ^ 63;
        print("x={}, y={}, z={}\n", .{ x, y, z });
    }
    //const x = highest_set_bit(@as(u64, 0));
    //const y = @clz(u64, @as(u64, 0)); // in a BE sense
    //const z = y ^ 63;
    //print("x={}, y={}, z={}\n", .{ x, y, z });
}

/// ostream_t/istream_t are used to wrap the pointers/length data passed into
/// ZSTD_decompress, so that all IO operations are safely bounds checked
/// They are written/read forward, and reads are treated as little-endian
/// They should be used opaquely to ensure safety
const ZStdOStream = struct {
    const Self = @This();

    // for debug
    magic: u32 = 0,
    // Current slice being processed
    pos: usize = 0,
    ptr: []u8 = undefined,

    /// Advances the stream by `len` bytes, and returns a pointer to the chunk that
    /// was skipped so it can be written to.
    fn get_write_ptr(s: *Self, len: usize) ![]u8 {
        if (len > (s.*.ptr.len - s.*.pos)) {
            return error.TooLittleInputSpace;
        }
        const slice = s.*.ptr[s.*.pos .. s.*.pos + len];
        s.*.pos += len;
        return slice;
    }

    /// Write the given byte into the output stream
    inline fn write_byte(s: *Self, symb: u8) !void {
        if (s.*.pos >= s.*.ptr.len) {
            print("outputting {} = {any}\n", .{ symb, s.*.ptr[0..s.*.pos] });
            return error.TooLittleOutputSpace;
        }
        
        s.*.ptr[s.*.pos] = symb;
        s.*.pos += 1;
    }

    /// Write the given bytes into the output stream
    inline fn write(s: *Self, input: []const u8) !void {
        if (input.len > (s.*.ptr.len - s.*.pos)) {
            return error.TooLittleOutputSpace;
        }
        for (input) |e| {
            s.*.ptr[s.*.pos] = e;
            s.*.pos += 1;
        }
    }

    /// Returns an `ostream_t` constructed from the given slice.
    fn from_slice(output: []u8) Self {
        return Self {.ptr = output };
    }
};

/// instream with bitreader
const ZStdIStream = struct {
    const Self = @This();

    /// 64-bit cache
    bit_cache: u64 = 0,
    /// Number of bits in cache (idx - bit_idx)
    in_cache: usize = 0,
    /// Input often reads a few bits at a time, so maintain bit offset/index
    idx: usize = 0,
    /// Real bit idx
    bit_idx: usize = 0,
    /// buffer slice (bit implicit length)
    buf: []const u8,

    inline fn fill_cache(s: *Self, num_bits: usize) !void {
        // Do we have space?
        assert(s.*.idx - s.*.bit_idx == s.*.in_cache);
        const available = s.*.bits_left();
        const to_read = @minimum(num_bits, available);
        var byte_pos = s.*.idx / 8;
        //print("fill_cache: available={}, to_read={}, num_bits={}, idx={}, bit_idx={}, byte_pos={}\n", .{
        //    available, to_read, num_bits, s.*.idx, s.*.bit_idx, byte_pos,
        //});
        while (s.*.in_cache < to_read) : (byte_pos += 1) {
            if (byte_pos >= s.*.buf.len) {
                return error.CacheTooLittleInputData;
            }
            s.*.bit_cache |= @as(u64, s.*.buf[byte_pos]) << @truncate(u6, s.*.in_cache);
            s.*.in_cache += 8;
            s.*.idx += 8;
        }
        // byte_pos = @minimum(s.*.idx / 8, s.*.buf.len - 1);
        // print("fill_cache: num_bits={}, idx={}, bit_idx={}, byte_pos={}\n", .{
        //    num_bits, s.*.idx, s.*.bit_idx, byte_pos,
        // });
        // print("fill_cache: num_bits={}, buf[{}]={x:02}, cache={x:016}/{}\n", .{
        //    num_bits, s.*.idx / 8, s.*.buf[byte_pos], s.*.bit_cache, s.*.in_cache,
        // });
    }

    // The following two functions are the only ones that allow the istream to be
    // non-byte aligned
    /// Reads `num` bits from a bitstream, and updates the internal offset
    fn read_bits(s: *Self, num_bits: usize) !u64 {
        if (num_bits > 64) {
            return error.TooManyBitsRequested;
        } else {
            if (num_bits > s.*.in_cache) {
                try s.*.fill_cache(num_bits);
            }
            const result = s.bit_cache & (U64_MAX >> @truncate(u6, 64 - num_bits));
            // drop num_bits
            s.*.bit_cache >>= @truncate(u6, num_bits);
            s.*.in_cache -= num_bits;
            s.*.bit_idx += num_bits;
            // print("num_bits={}, result={x}, in_cache={}, idx={}\n", .{ num_bits, result, s.*.in_cache, s.*.idx });
            return result;
        }
    }

    /// Backs-up the stream by `num` bits so they can be read again
    /// If a non-zero number of bits have been read from the current byte, advance
    /// the offset to the next byte
    fn rewind_bits(s: *Self, num_bits: usize) !void {
        if (num_bits > s.*.bit_idx) {
            return error.TooManyBitsReturned;
        }
        // Must move back to bit_idx
        s.*.bit_idx -= num_bits;
        s.*.idx = s.bit_idx;
        // How many bits must we drop from current byte?
        const drop_bits = s.*.bit_idx % 8;
        s.*.bit_cache = @as(u64, s.*.buf[s.*.bit_idx / 8]);
        s.*.bit_cache >>= @truncate(u6, drop_bits);
        s.*.in_cache = 8 - drop_bits;
        s.*.idx += s.*.in_cache;
    }

    /// Advance the inner state by `len` bytes.  The stream must be byte aligned.
    inline fn advance(s: *Self, num_bytes: usize) !void {
        var num_bits = num_bytes * 8;
        while (num_bits > 32) : (num_bits -= 32) {
            _ = try s.read_bits(32);
        }
        if (num_bits > 0) {
            _ = try s.read_bits(num_bits);
        }
        // if (len > in->len) {
        //      INP_SIZE();
        // }
        // if (in->bit_offset != 0) {
        //     ERROR("Attempting to operate on a non-byte aligned stream");
        // }
        // in->ptr += len;
        // in->len -= len;
    }


    /// Returns the number of bytes left to be read in this stream.  The stream must
    /// be byte aligned.
    inline fn length(s: *const Self) usize {
        return s.buf.len - (s.*.bit_idx / 8);
    }

    inline fn bits_left(s: *const Self) usize {
        return (s.*.buf.len * 8) - s.*.bit_idx;
    }


    /// If the remaining bits in a byte will be unused, advance to the end of the
    /// byte
    fn align_stream(s: *Self) !void {
        const aligned = 8 - (s.*.bit_idx % 8);
        print("align_stream: idx={}, bit_idx={}, aligned={}\n", .{ s.*.idx, s.*.bit_idx, aligned });
        if (s.*.bit_idx % 8 != 0) {
            if (s.*.buf.len == 0) {
                // INP_SIZE();
                return error.AlignTooLittleInputData;
            }
            s.*.bit_idx += aligned;//in->bit_offset = 0;
            s.*.idx = s.*.bit_idx;
            s.*.in_cache = 0;
            s.*.bit_cache = 0;
        }
    }

    /// Returns a pointer where `len` bytes can be read, and advances the internal
    /// state.  The stream must be byte aligned.
    /// We return a slice
    fn get_read_ptr(s: *Self, len: usize) ![]const u8 {
        const real_idx = s.*.bit_idx;
        const byte_idx = (real_idx / 8);
        const bytes_left = s.*.buf.len - byte_idx;
        if ((real_idx % 8) != 0) {
            print("Attempting to operate on a non-byte aligned stream\n", .{});
            print("real={}, byte_idx={}, bytes_left={}, idx={}/{}\n", .{
                real_idx, byte_idx, bytes_left, s.*.idx, s.*.idx % 8
            });
            return error.InputNotByteAligned;
        }
        if (len > bytes_left) {
            print("byte_idx={}, bytes_left={}, len={}\n", .{ byte_idx, bytes_left, len });
            return error.TooLittleInputData;
        }
        const slice = s.*.buf[byte_idx .. byte_idx + len];
        s.*.idx = real_idx + len * 8;
        s.*.bit_idx = s.*.idx;
        s.*.in_cache = 0;
        s.*.bit_cache = 0;
        return slice;
    }

    /// Returns an `istream_t` with the same base as `in`, and length `len`
    /// Then, advance `in` to account for the consumed bytes
    /// `in` must be byte aligned
    fn make_sub_istream(s: *Self, len: usize) !Self {
        // Consume `len` bytes of the parent stream
        const slice = try s.*.get_read_ptr(len);
        assert(slice.len == len);
        // Make a substream using the pointer to those `len` bytes
        return Self { .buf = slice };
    }

    inline fn from_slice(input: []const u8) Self {
        return Self { .buf = input };
    }
};

/// Read `num` bits (up to 64) from `src + offset`, where `offset` is in bits,
/// and return them interpreted as a little-endian unsigned integer.
inline fn read_bits_LE(src: []const u8,
                pnum_bits: i64,
                offset: usize) !u64 {
    const num_bits = @bitCast(u64, pnum_bits);
    // const maxbits = (src.len * 8) - offset;

    if (pnum_bits > 64) {
        return error.InvalidNumberOfBits;
    }

    if (pnum_bits < 1) {
        return 0;
    }

    // Skip over bytes that aren't in range
    var ofs = (offset >> 0x3);
    var bit_offset = @truncate(u6, offset % 8);
    // var pos = (offset / 8);
    var res: u64 = 0;
    // var shift: u6 = 0;
    // var left: i64 = @bitCast(i64, num_bits);
    // This may be parts of multiple bytes
    var bytes = ((7 + num_bits + bit_offset) / 8);

    const mask = @as(u64, std.math.maxInt(u64)) >> @truncate(u6, 64 - num_bits);
    // print("maxbits={}, ofs={}, len={}, pos={}, bit_offset={}, left={}, num_bits={}/{}, bytes={}, mask={x}\n", .{
    //     maxbits, ofs, src.len, pos, bit_offset, left, num_bits, pnum_bits, bytes, mask,
    // });

    for (src[ofs..ofs + bytes]) |e, i| {
        // Read the next byte, shift it to account for the offset, and then mask
        // out the top part if we don't need all the bits
        res += (@as(u64, e) << @truncate(u6, 8 * i));
        // print("> res={x}\n", .{ res });
        //pos -%= 1;
        bytes -= 1;
    }
    return (res >> bit_offset) & mask;
}

inline fn i64tou64(value: i64) u64 {
    //return if (value < 0) @bitCast(u64, -value) else @bitCast(u64, value);
    return @bitCast(u64, if (value < 0) -value else value);
}

/// Read bits from the end of a HUF or FSE bitstream.  `offset` is in
/// bits, so it updates `offset` to `offset - bits`, and then reads
/// `bits` bits from `src + offset`.  If the offset becomes negative,
/// the extra bits at the bottom are filled in with `0` bits instead
/// of reading from before `src`.
fn STREAM_read_bits(src: []const u8,
                    bits: i64,
                    offset: *i64) !u64 {
    var loffset = offset.*;
    var lbits = bits;
    offset.* = offset.* - bits;
    loffset = loffset - lbits;
    var actual_bits: u64 = i64tou64(bits);
    var actual_off: u64 = i64tou64(offset.*);
    // print("{}/actual_bits={}, {}/actual_off={}\n", .{
    //     bits, actual_bits, offset.* + bits, actual_off,
    // });
    // Don't actually read bits from before the start of src, so if `*offset <
    // 0` fix actual_off and actual_bits to reflect the quantity to read

    // print("=> {}: bits={}, actual_bits={}, actual_off={}, offset={}/{}\n", .{
    //     @This(),
    //     bits, actual_bits, actual_off, offset.*, loffset,
    // });
    if (offset.* < 0) {
        //print("actual_bits={}\n", .{ actual_bits });
        lbits = bits + loffset;
        actual_bits = i64tou64(lbits);
        actual_off = 0;
        //print("actual_bits={}\n", .{ actual_bits });
    }
    var res: u64 = 0;
    // print("== STREAM_read_bits: bits={}, actual_bits={}, actual_off={}, offset={}\n", .{
    //     bits, actual_bits, actual_off, offset.*,
    // });

    if (true) {
        res = try read_bits_LE(src, lbits, actual_off);
        // print("res=0x{x:0<16}\n", .{ res, });
        if (offset.* < 0) {
            // Fill in the bottom "overflowed" bits with 0's
            // res = -*offset >= 64 ? 0 : (res << -*offset);
            res = if (-(offset.*) >= 64) 0 else (res << (@truncate(u6, @bitCast(u64, -(offset.*)))));
        }
    }
    // print("<= STREAM_read_bits: res=0x{x:0<16}, offset={}\n", .{ res, offset.* });

    return res;
}
//*** END IO STREAM OPERATIONS

// HUFFMAN PRIMITIVES
/// Table decode method uses exponential memory, so we need to limit depth
const HUF_MAX_BITS = 16;

/// Limit the maximum number of symbols to 256 so we can store a symbol in a byte
const HUF_MAX_SYMBS = 256;

/// Structure containing all tables necessary for efficient Huffman decoding
const HUFTable = struct {
    const Self = @This();

    symbols: ?[]u8 = null,
    num_bits: ?[]u8 = null,
    max_bits: i64 = 0,

    /// Initializes a Huffman table using canonical Huffman codes
    /// For more explanation on canonical Huffman codes see
    /// http://www.cs.uofs.edu/~mccloske/courses/cmps340/huff_canonical_dec2015.html
    /// Codes within a level are allocated in symbol order (i.e. smaller symbols get
    /// earlier codes)
    /// Initialize a Huffman decoding table using the table of bit counts provided
    fn init_dtable(s: *Self, bits: []const u8,
                   num_symbs: i64, allocator: *Allocator) !void {
        // memset(s, 0, sizeof(HUFTable));
        if (num_symbs > HUF_MAX_SYMBS) {
            // ERROR("Too many symbols for Huffman");
            return error.TooManySymbolsForHuffman;
        }

        var max_bits: u8 = 0;
        var rank_count: [HUF_MAX_BITS + 1]u16 = [_]u16{0} ** (HUF_MAX_BITS + 1);

        // Count the number of symbols for each number of bits, and determine the
        // depth of the tree
        const max_symbs = @bitCast(u64, num_symbs);
        for (bits) |e| {
            if (e > HUF_MAX_BITS) {
                return error.HuffmanTableDepthTooLarge;
            }
            max_bits = @maximum(max_bits, e);
            rank_count[e] += 1;
        }

        const table_size: usize = @as(u64, 1) << @truncate(u6, max_bits);
        print(@src().fn_name ++ ": table_size={}, max_bits={}\n", .{ table_size, max_bits, });
        s.*.max_bits = max_bits;
        s.*.symbols = try allocator.*.alloc(u8, table_size);
        s.*.num_bits = try allocator.*.alloc(u8, table_size);

        // @TODO: check allocations
        // if (!table->symbols || !table->num_bits) {
        //     free(table->symbols);
        //     free(table->num_bits);
        //     BAD_ALLOC();
        // }

        // "Symbols are sorted by Weight. Within same Weight, symbols keep natural
        // order. Symbols with a Weight of zero are removed. Then, starting from
        // lowest weight, prefix codes are distributed in order."

        var rank_idx: [HUF_MAX_BITS + 1]u32 = [_]u32{0} ** (HUF_MAX_BITS + 1);
        // Initialize the starting codes for each rank (number of bits)
        rank_idx[max_bits] = 0;
        {
            var i: u8 = max_bits;
            // for (int i = max_bits; i >= 1; i--) {
            while (i >= 1) : (i -= 1) {
                rank_idx[i - 1] = rank_idx[i] + rank_count[i] * (@as(u32, 1) << @truncate(u5, max_bits - i));
                // The entire range takes the same number of bits so we can memset it
                // memset(&table->num_bits[rank_idx[i]], i, rank_idx[i - 1] - rank_idx[i]);
                print("rank idx[{}]: {}..{}\n", .{ i, rank_idx[i], rank_idx[i - 1] });
                for (s.*.num_bits.?[rank_idx[i]..rank_idx[i - 1]]) |*e| {
                    e.* = i;
                }
            }
        }

        if (rank_idx[0] != table_size) {
            print(@src().fn_name ++ ": Corruption rank_idx[0]={}, table_size={}\n",
                  .{ rank_idx[0], table_size} );
            return error.Corruption;
        }

        // Allocate codes and fill in the table
        {
            var i: usize = 0;
            // for (int i = 0; i < num_symbs; i++) {
            while (i < max_symbs) : (i += 1) {
                if (bits[i] != 0) {
                    // Allocate a code for this symbol and set its range in the table
                    const code = rank_idx[bits[i]];
                    // Since the code doesn't care about the bottom `max_bits - bits[i]`
                    // bits of state, it gets a range that spans all possible values of
                    // the lower bits
                    const len: u16 = @as(u16, 1) << @truncate(u4, max_bits - bits[i]);
                    // memset(&table->symbols[code], i, len);
                    const ii = @truncate(u8, i); 
                    for (s.*.symbols.?[code..code+len]) |*e| {
                        e.* = ii;
                    }
                    rank_idx[bits[i]] += len;
                }
            }
        }
    }

    /// Initialize a Huffman decoding table using the table of weights provided
    /// Weights follow the definition provided in the Zstandard specification
    fn init_usingweights(s: *HUFTable, weights: []const u8, num_symbs: i64,
                         allocator: *Allocator) !void {
        // +1 because the last weight is not transmitted in the header
        if ((num_symbs + 1) > HUF_MAX_SYMBS) {
            print(@src().fn_name ++ ": nsymbols={}", .{ num_symbs + 1});
            return error.TooManySymbolsForHuffman;
        }

        var bits: [HUF_MAX_SYMBS]u8 = [_]u8{0} ** HUF_MAX_SYMBS ;
        var weight_sum: u64 = 0;
        const max_symbs: usize = @bitCast(usize, num_symbs);
        var i: usize = 0;
        while (i < max_symbs) : (i += 1) {
            // Weights are in the same range as bit count
            if (weights[i] > HUF_MAX_BITS) {
                return error.Corruption;
            }
            weight_sum += if (weights[i] > 0) @as(u64, 1) << @truncate(u6, weights[i] - 1) else 0;
        }

        // Find the first power of 2 larger than the sum
        const max_bits = @bitCast(u16, highest_set_bit(weight_sum)) + 1;
        print(@src().fn_name ++ ": num_symbs={}, weight_sum={}, max_bits={}\n",
              .{ num_symbs, weight_sum, max_bits });
        const left_over: u64 = (@as(u64, 1) << @truncate(u6, max_bits)) - weight_sum;
        // If the left over isn't a power of 2, the weights are invalid
        if ((left_over & (left_over - 1)) != 0) {
            return error.Corruption;
        }
        
        // left_over is used to find the last weight as it's not transmitted
        // by inverting 2^(weight - 1) we can determine the value of last_weight
        const last_weight = @bitCast(u16, highest_set_bit(left_over) + 1);
        
        i = 0;
        while (i < max_symbs) : (i += 1) {
            // "Number_of_Bits = Number_of_Bits ? Max_Number_of_Bits + 1 - Weight : 0"
            bits[i] = if (weights[i] > 0) (@truncate(u8, max_bits + 1) - weights[i]) else 0;
        }
        bits[max_symbs] = @truncate(u8, max_bits + 1) - @truncate(u8, last_weight); // Last weight is always non-zero

        try s.init_dtable(bits[0..], num_symbs + 1, allocator);
    }

    /// Read in a full state's worth of bits to initialize the state
    inline fn init_state(s: *Self, state: *u16, src: []const u8, offset: *i64) !void {
        // Read in a full `dtable->max_bits` bits to initialize the state
        state.* = @truncate(u16, try STREAM_read_bits(src, s.*.max_bits, offset));
    }

    /// Decode a single symbol and read in enough bits to refresh the state
    fn decode_symbol(s: *Self, state: *u16, src: []const u8, offset: *i64) !u8 {
        // Look up the symbol and number of bits to read
        const symb = s.*.symbols.?[state.*];
        const bits = s.*.num_bits.?[state.*];
        const rest = try STREAM_read_bits(src, bits, offset);
        // Shift `bits` bits out of the state, keeping the low order bits that
        // weren't necessary to determine this symbol.  Then add in the new bits
        // read from the stream.
        const lmax_bits = @truncate(u4, @bitCast(u64, s.*.max_bits));
        state.* = ((state.* << @truncate(u4, bits)) + @truncate(u16, rest)) &
            ((@as(u16, 1) << lmax_bits) - 1);

        return symb;
    }
    
    /// Decompresses a single Huffman stream, returns the number of bytes decoded.
    /// `src_len` must be the exact length of the Huffman-coded block.
    fn decompress_1stream(s: *Self, out: *ZStdOStream, in: *ZStdIStream) !usize {
        const len = in.*.length();
        if (len == 0) {
            return error.BadInputSize;
        }
        const src = try in.get_read_ptr(len);

        // "Each bitstream must be read backward, that is starting from the end down
        // to the beginning. Therefore it's necessary to know the size of each
        // bitstream.
        //
        // It's also necessary to know exactly which bit is the latest. This is
        // detected by a final bit flag : the highest bit of latest byte is a
        // final-bit-flag. Consequently, a last byte of 0 is not possible. And the
        // final-bit-flag itself is not part of the useful bitstream. Hence, the
        // last byte contains between 0 and 7 useful bits."
        const padding = 8 - highest_set_bit(src[len - 1]);

        // Offset starts at the end because HUF streams are read backwards
        var bit_offset: i64 = @bitCast(i64, len * 8) - padding;
        var state: u16 = 0;

        try s.*.init_state(&state, src, &bit_offset);

        var symbols_written: usize = 0;
        while (bit_offset > -s.*.max_bits) {
            // Iterate over the stream, decoding one symbol at a time
            const symbol = try s.*.decode_symbol(&state, src, &bit_offset);
            try out.*.write_byte(symbol);
            // print("decompress_1stream: bit_offset={}, symbol={c}\n", .{ bit_offset, symbol });
            symbols_written += 1;
        }
        // "The process continues up to reading the required number of symbols per
        // stream. If a bitstream is not entirely and exactly consumed, hence
        // reaching exactly its beginning position with all bits consumed, the
        // decoding process is considered faulty."

        // When all symbols have been decoded, the final state value shouldn't have
        // any data from the stream, so it should have "read" dtable->max_bits from
        // before the start of `src`
        // Therefore `offset`, the edge to start reading new bits at, should be
        // dtable->max_bits before the start of the stream
        // print("decompress_1stream: bit_offset={}, max_bits={}, symbols_written={}, len={}\n", .{
        //     bit_offset, s.*.max_bits, symbols_written, len,
        // });

        if (bit_offset != -s.*.max_bits) {
            print(@src().fn_name ++ ": bit_offset={} != -s.*.max_bits={}", .{ bit_offset, -s.*.max_bits });
            return error.Corruption;
        }

        return symbols_written;
    }

    /// Same as previous but decodes 4 streams, formatted as in the Zstandard
    /// specification.
    /// `src_len` must be the exact length of the Huffman-coded block.
    fn decompress_4stream(s: *Self, out: *ZStdOStream, in: *ZStdIStream) !usize {
        // "Compressed size is provided explicitly : in the 4-streams variant,
        // bitstreams are preceded by 3 unsigned little-endian 16-bits values. Each
        // value represents the compressed size of one stream, in order. The last
        // stream size is deducted from total compressed size and from previously
        // decoded stream sizes"
        const csize1 = try in.*.read_bits(16);
        const csize2 = try in.*.read_bits(16);
        const csize3 = try in.*.read_bits(16);

        var in1 = try in.*.make_sub_istream(csize1);
        var in2 = try in.*.make_sub_istream(csize2);
        var in3 = try in.*.make_sub_istream(csize3);
        const csize4 = in.*.length();
        var in4 = try in.*.make_sub_istream(csize4);
        print(@src().fn_name ++ ": csize1={}, csize2={}, csize3={}, csize4={}\n",
              .{ csize1, csize2, csize3, csize4, });

        var total_output: usize = 0;
        // Decode each stream independently for simplicity
        // If we wanted to we could decode all 4 at the same time for speed,
        // utilizing more execution units
        total_output += try s.*.decompress_1stream(out, &in1);
        total_output += try s.*.decompress_1stream(out, &in2);
        total_output += try s.*.decompress_1stream(out, &in3);
        total_output += try s.*.decompress_1stream(out, &in4);

        print(@src().fn_name ++ ": total_output={}\n", .{ total_output});

        return total_output;
    }

    
    fn free_dtable(s: *Self, allocator: *Allocator) void {
        s.*.max_bits = 0;
        // DANGER
        if (s.*.symbols) |symbols| {
            allocator.*.free(symbols);
            s.*.symbols = null;
        }
        if (s.*.num_bits) |num_bits| {
            allocator.*.free(num_bits);
            s.*.num_bits = null;
        }
    }
};


// /// Deep copy a decoding table, so that it can be used and free'd without
// /// impacting the source table.
// static void HUF_copy_dtable(HUFTable *const dst, const HUFTable *const src);
// static void HUF_copy_dtable(HUFTable *const dst,
//                             const HUFTable *const src) {
//     if (src->max_bits == 0) {
//         memset(dst, 0, sizeof(HUFTable));
//         return;
//     }

//     const size_t size = (size_t)1 << src->max_bits;
//     dst->max_bits = src->max_bits;

//     dst->symbols = malloc(size);
//     dst->num_bits = malloc(size);
//     if (!dst->symbols || !dst->num_bits) {
//         BAD_ALLOC();
//     }

//     memcpy(dst->symbols, src->symbols, size);
//     memcpy(dst->num_bits, src->num_bits, size);
// }
//*** END HUFFMAN PRIMITIVES


//*** FSE PRIMITIVES
// For more description of FSE see
// https://github.com/Cyan4973/FiniteStateEntropy/


/// The tables needed to decode FSE encoded streams
const FSETable = struct {
    const Self = @This();

    /// FSE table decoding uses exponential memory, so limit the maximum accuracy
    const FSE_MAX_ACCURACY_LOG = 15;
    /// Limit the maximum number of symbols so they can be stored in a single byte
    const FSE_MAX_SYMBS = 256;

    symbols: ?[]u8 = null,
    num_bits: ?[]u8 = null,
    new_state_base: ?[]u16 = null,
    accuracy_log: u8 = 0,

    /// Read bits from the stream to initialize the state and shift offset back
    // static inline void FSE_init_state(const FSETable *const dtable,
    //                                   u16 *const state, const u8 *const src,
    //                                   i64 *const offset);
    inline fn init_state(s: *Self, state: *u16, src: []const u8, offset: *i64) !void {
        // Read in a full `accuracy_log` bits to initialize the state
        const bits: u8 = s.*.accuracy_log;
        state.* = @truncate(u16, try STREAM_read_bits(src, bits, offset));
    }

    fn free_dtable(s: *Self, allocator: *Allocator) void {
        s.*.accuracy_log = 0;
        if (s.*.symbols) |ptr| {
            allocator.*.free(ptr);
            s.*.symbols = null;
        }
        if (s.*.num_bits) |ptr| {
            allocator.*.free(ptr);
            s.*.num_bits = null;
        }
        if (s.*.new_state_base) |ptr| {
            allocator.*.free(ptr);
            s.*.new_state_base = null;
        }
    }

    /// Return the symbol for the current state
    inline fn peek_symbol(s: *const Self, state: u16) !u8 {
        if (s.*.symbols) |symbols| {
            if (state < symbols.len) {
                return symbols[state];
            } else {
                return error.StateOutOfRange;
            }
        } else {
            return error.SymbolsIsNull;
        }
    }

    /// Decodes a single FSE symbol and updates the offset
    /// Combine peek and update: decode a symbol and update the state
    inline fn decode_symbol(s: *Self, state: *u16, src: []const u8, offset: *i64) !u8 {
        const symb = s.*.peek_symbol(state.*);
        try s.*.update_state(state, src, offset);
        return symb;
    }

    /// Consumes bits from the input and uses the current state to determine the
    /// next state
    /// Read the number of bits necessary to update state, update, and shift offset
    /// back to reflect the bits read
    inline fn update_state(s: *Self, state: *u16, src: []const u8,
                               offset: *i64) !void {
        const bits = s.*.num_bits.?[state.*];
        const rest: u16 = @truncate(u16, try STREAM_read_bits(src, bits, offset));
        state.* = s.*.new_state_base.?[state.*] + rest;
    }
};


//******* FSE PRIMITIVES
// For more description of FSE see
// https://github.com/Cyan4973/FiniteStateEntropy/

/// Decompress two interleaved bitstreams (e.g. compressed Huffman weights)
/// using an FSE decoding table.  `src_len` must be the exact length of the
/// block.
fn FSE_decompress_interleaved2(dtable: *FSETable,
                               out: *ZStdOStream,
                               in: *ZStdIStream) !i64 {
    const len = in.*.length();
    if (len == 0) {
        return error.InputSize;
    }
    const src = try in.*.get_read_ptr(len);

    // "Each bitstream must be read backward, that is starting from the end down
    // to the beginning. Therefore it's necessary to know the size of each
    // bitstream.
    //
    // It's also necessary to know exactly which bit is the latest. This is
    // detected by a final bit flag : the highest bit of latest byte is a
    // final-bit-flag. Consequently, a last byte of 0 is not possible. And the
    // final-bit-flag itself is not part of the useful bitstream. Hence, the
    // last byte contains between 0 and 7 useful bits."
    //     const int padding = 8 - highest_set_bit(src[len - 1]);
    //     i64 offset = len * 8 - padding;
    const padding = 8 - highest_set_bit(src[len - 1]);
    const upadding = @bitCast(u16, padding);
    // The offset starts at the end because FSE streams are read backwards
    var offset = @bitCast(i64, (len * 8) - upadding);

    var state1: u16 = 0;
    var state2: u16 = 0;
    // "The first state (State1) encodes the even indexed symbols, and the
    // second (State2) encodes the odd indexes. State1 is initialized first, and
    // then State2, and they take turns decoding a single symbol and updating
    // their state."
    try dtable.init_state(&state1, src, &offset);
    try dtable.init_state(&state2, src, &offset);

    // Decode until we overflow the stream
    // Since we decode in reverse order, overflowing the stream is offset going
    // negative
    var symbols_written: i64 = 0;
    while (true) {
        if (debug) {
            print(@src().fn_name ++ ": symbols_written={}, offset={}\n",
                  .{ symbols_written, offset } );
        }
        // "The number of symbols to decode is determined by tracking bitStream
        // overflow condition: If updating state after decoding a symbol would
        // require more bits than remain in the stream, it is assumed the extra
        // bits are 0. Then, the symbols for each of the final states are
        // decoded and the process is complete."
        try out.*.write_byte(try dtable.*.decode_symbol(&state1, src[0..], &offset));
        symbols_written += 1;
        if (offset < 0) {
            // There's still a symbol to decode in state2
            try out.*.write_byte(try dtable.*.peek_symbol(state2));
            symbols_written += 1;
            break;
        }

        try out.*.write_byte(try dtable.*.decode_symbol(&state2, src, &offset));
        symbols_written += 1;
        if (offset < 0) {
            // There's still a symbol to decode in state1
            try out.*.write_byte(try dtable.*.peek_symbol(state1));
            symbols_written += 1;
            break;
        }
    }

    return symbols_written;
}

/// Initialize a decoding table using normalized frequencies.
fn FSE_init_dtable(dtable: *FSETable,
                   norm_freqs: []const i16, num_symbs: usize,
                   accuracy_log: u8,
                   allocator: *Allocator) !void {
    if (accuracy_log > FSETable.FSE_MAX_ACCURACY_LOG) {
        // ERROR("FSE accuracy too large");
        return error.FSEAccuracyTooLarge;
    }
    if (num_symbs > FSETable.FSE_MAX_SYMBS) {
        // ERROR("Too many symbols for FSE");
        return error.TooManySymbols;
    }

    dtable.*.accuracy_log = accuracy_log;

    const size: usize = @as(usize, 1) << @truncate(u6, accuracy_log);
    print(@src().fn_name ++ ": Must alloca {}/{} items for FSETable\n", .{ size, accuracy_log, });
    if ( dtable.*.symbols) |ptr| {
        allocator.*.free(ptr);
        dtable.*.symbols = null;
    }
    dtable.*.symbols = try allocator.*.alloc(u8, size);
    if ( dtable.*.num_bits) |ptr| {
        allocator.*.free(ptr);
        dtable.*.num_bits = null;
    }
    dtable.*.num_bits = try allocator.*.alloc(u8, size);
    if ( dtable.*.new_state_base) |ptr| {
        allocator.*.free(ptr);
        dtable.*.new_state_base = null;
    }
    dtable.*.new_state_base = try allocator.*.alloc(u16, size);

    //     if (!dtable->symbols || !dtable->num_bits || !dtable->new_state_base) {
    //         BAD_ALLOC();
    //     }

    // Used to determine how many bits need to be read for each state,
    // and where the destination range should start
    // Needs to be u16 because max value is 2 * max number of symbols,
    // which can be larger than a byte can store
    var state_desc: [FSETable.FSE_MAX_SYMBS]u16 = [_]u16{0} ** FSETable.FSE_MAX_SYMBS;

    // "Symbols are scanned in their natural order for "less than 1"
    // probabilities. Symbols with this probability are being attributed a
    // single cell, starting from the end of the table. These symbols define a
    // full state reset, reading Accuracy_Log bits."
    var high_threshold = size;
    var idx: usize = 0;
    //     for (int s = 0; s < num_symbs; s++) {
    print(@src().fn_name ++ ": norm_freqs.len={}, num_symbs={}\n", .{ norm_freqs.len, num_symbs, });
    while (idx < num_symbs) : (idx += 1) {
        // Scan for low probability symbols to put at the top
        if (norm_freqs[idx] == -1) {
            high_threshold -= 1;
            dtable.*.symbols.?[high_threshold] = @truncate(u8, idx);
            state_desc[idx] = 1;
        }
    }

    // "All remaining symbols are sorted in their natural order. Starting from
    // symbol 0 and table position 0, each symbol gets attributed as many cells
    // as its probability. Cell allocation is spreaded, not linear."
    // Place the rest in the table
    const step = @truncate(u16, (size >> 1) + (size >> 3) + 3);
    const mask = @truncate(u16, size - 1);
    print(@src().fn_name ++ ": step={}, mask={}\n", .{ step, mask, });
    var pos: u16 = 0;
    idx = 0;
    while (idx < num_symbs) : (idx += 1) {
        if (norm_freqs[idx] <= 0) {
            continue;
        }

        // Oh crap...u16 vs i16
        state_desc[idx] = @intCast(u16, norm_freqs[idx]);

        var i: usize = 0;
        while (i < norm_freqs[idx]) : (i += 1) {
            // Give `norm_freqs[s]` states to symbol s
            dtable.*.symbols.?[pos] = @truncate(u8, idx);
            // "A position is skipped if already occupied, typically by a "less
            // than 1" probability symbol."
            inner: while (true) {
                pos = (pos + step) & mask;
                if (pos < high_threshold) {
                    break :inner;
                }
            }
            // Note: no other collision checking is necessary as `step` is
            // coprime to `size`, so the cycle will visit each position exactly
            // once
        }
    }
    if (pos != 0) {
        return error.PosCorruption;
    }

    // Now we can fill baseline and num bits
    var iter: usize = 0;
    // for (size_t i = 0; i < size; i++) {
    while (iter < size) : (iter += 1) {
        const symbol = dtable.*.symbols.?[iter];
        const next_state_desc = state_desc[symbol];
        state_desc[symbol] += 1;
        // Fills in the table appropriately, next_state_desc increases by symbol
        // over time, decreasing number of bits
        const highest_bit = highest_set_bit(next_state_desc);
        // print("next_state_desc={x}, highest_bit={}, accuracy_log={}\n", .{
        //     next_state_desc, highest_bit, accuracy_log,
        // });
        dtable.*.num_bits.?[iter] = @truncate(u8, accuracy_log - @bitCast(u16, highest_bit));
        // Baseline increases until the bit threshold is passed, at which point
        // it resets to 0
        dtable.*.new_state_base.?[iter] =
            (next_state_desc << @truncate(u4, dtable.*.num_bits.?[iter])) -% @truncate(u16, size);
    }
}

/// Decode an FSE header as defined in the Zstandard format specification and
/// use the decoded frequencies to initialize a decoding table.
fn FSE_decode_header(dtable: *FSETable, in: *ZStdIStream,
                     max_accuracy_log: u6, allocator: *Allocator) !void {
    // "An FSE distribution table describes the probabilities of all symbols
    // from 0 to the last present one (included) on a normalized scale of 1 <<
    // Accuracy_Log .
    //
    // It's a bitstream which is read forward, in little-endian fashion. It's
    // not necessary to know its exact size, since it will be discovered and
    // reported by the decoding process.
    if (max_accuracy_log > FSETable.FSE_MAX_ACCURACY_LOG) {
        print(@src().fn_name ++ ": max_accuracy_log={}, FSE_MAX_ACCURACY_LOG={}\n",
              .{ max_accuracy_log, FSETable.FSE_MAX_ACCURACY_LOG, });
        return error.FSEAccuracyTooLarge;
    }

    // The bitstream starts by reporting on which scale it operates.
    // Accuracy_Log = low4bits + 5. Note that maximum Accuracy_Log for literal
    // and match lengths is 9, and for offsets is 8. Higher values are
    // considered errors."
    const accuracy_log: u6 = 5 + @truncate(u6, try in.*.read_bits(4));
    if (accuracy_log > max_accuracy_log) {
        print(@src().fn_name ++ ": max_accuracy_log={}, accuracy_log={}\n",
              .{ max_accuracy_log, accuracy_log, });
        return error.FSEAccuracyTooLarge;
    }

    // "Then follows each symbol value, from 0 to last present one. The number
    // of bits used by each field is variable. It depends on :
    //
    // Remaining probabilities + 1 : example : Presuming an Accuracy_Log of 8,
    // and presuming 100 probabilities points have already been distributed, the
    // decoder may read any value from 0 to 255 - 100 + 1 == 156 (inclusive).
    // Therefore, it must read log2sup(156) == 8 bits.
    //
    // Value decoded : small values use 1 less bit : example : Presuming values
    // from 0 to 156 (inclusive) are possible, 255-156 = 99 values are remaining
    // in an 8-bits field. They are used this way : first 99 values (hence from
    // 0 to 98) use only 7 bits, values from 99 to 156 use 8 bits. "

    var remaining: i32 = @bitCast(i32, @as(u32, 1) << @truncate(u5, accuracy_log));
    var frequencies: [FSETable.FSE_MAX_SYMBS]i16 = [_]i16{-1} ** FSETable.FSE_MAX_SYMBS;

    var symb: usize = 0;
    while ((remaining > 0) and (symb < FSETable.FSE_MAX_SYMBS)) {
        // Log of the number of possible values we could read
        const lremaining = @bitCast(u32, remaining);
        var bits = highest_set_bit(lremaining + 1) + 1;
        const ubits = @bitCast(u16, bits);
        var val: u16 = @truncate(u16, try in.*.read_bits(ubits));

        // Try to mask out the lower bits to see if it qualifies for the "small
        // value" threshold
        const lower_mask: u16 = (@as(u16, 1) << @truncate(u4, ubits - 1)) - 1;
        const threshold: u16 = (@as(u16, 1) << @truncate(u4, ubits)) - 1 - @truncate(u16, lremaining + 1);

        print(@src().fn_name ++ ": symb={}, remaining={}, bits={}/{}, val={}, lower_mask={}, threshold={}\n", .{
            symb, remaining, bits, ubits, val, lower_mask, threshold,
        });
        if ((val & lower_mask) < threshold) {
            try in.*.rewind_bits(1);
            val = val & lower_mask;
        } else if (val > lower_mask) {
            val = val - threshold;
        }

        // "Probability is obtained from Value decoded by following formula :
        // Proba = value - 1"
        const proba: i16 = @bitCast(i16, val) - 1;

        // "It means value 0 becomes negative probability -1. -1 is a special
        // probability, which means "less than 1". Its effect on distribution
        // table is described in next paragraph. For the purpose of calculating
        // cumulated distribution, it counts as one."
        remaining -= if (proba < 0) -proba else proba;
        print(@src().fn_name ++ ": remaining={}, proba={}\n", .{ remaining, proba });

        frequencies[symb] = proba;
        symb += 1;

        // "When a symbol has a probability of zero, it is followed by a 2-bits
        // repeat flag. This repeat flag tells how many probabilities of zeroes
        // follow the current one. It provides a number ranging from 0 to 3. If
        // it is a 3, another 2-bits repeat flag follows, and so on."
        if (proba == 0) {
            // Read the next two bits to see how many more 0s
            var repeat = try in.*.read_bits(2);

            while (true) {
                var i: u64 = 0;
                while ((i < repeat) and (symb < FSETable.FSE_MAX_SYMBS)) : (i += 1) {
                    frequencies[symb] = 0;
                    symb += 1;
                }
                if (repeat == 3) {
                    repeat = try in.*.read_bits(2);
                } else {
                    break;
                }
            }
        }
    }
    if (false) {
        for (frequencies) |e, i| {
            print("frequencies[{}]={}\n", .{ i, e });
        }
    }

    try in.*.align_stream();

    // "When last symbol reaches cumulated total of 1 << Accuracy_Log, decoding
    // is complete. If the last symbol makes cumulated total go above 1 <<
    // Accuracy_Log, distribution is considered corrupted."
    if ((remaining != 0) or (symb >= FSETable.FSE_MAX_SYMBS)) {
        // CORRUPTION();
        print(@src().fn_name ++ ": remaining={}, symb={}\n",
              .{remaining, symb});
        return error.Corruption;
    }

    // Initialize the decoding table using the determined weights
    try FSE_init_dtable(dtable, frequencies[0..], symb, accuracy_log, allocator);
}

/// Initialize an FSE table that will always return the same symbol and consume
/// 0 bits per symbol, to be used for RLE mode in sequence commands
fn FSE_init_dtable_rle(dtable: *FSETable, symbol: u8, allocator: *Allocator) !void {
    if (dtable.*.symbols) |ptr| {
        allocator.*.free(ptr);
        dtable.*.symbols = null;
    }
    dtable.*.symbols = try allocator.*.alloc(u8, 1);
    if (dtable.*.num_bits) |ptr| {
        allocator.*.free(ptr);
        dtable.*.num_bits = null;
    }
    dtable.*.num_bits = try allocator.*.alloc(u8, 1);
    if (dtable.*.new_state_base) |ptr| {
        allocator.*.free(ptr);
        dtable.*.new_state_base = null;
    }
    dtable.*.new_state_base = try allocator.*.alloc(u16, 1);

    // if (!dtable->symbols || !dtable->num_bits || !dtable->new_state_base) {
    //     BAD_ALLOC();
    // }

    // This setup will always have a state of 0, always return symbol `symb`,
    // and never consume any bits
    dtable.*.symbols.?[0] = symbol;
    dtable.*.num_bits.?[0] = 0;
    dtable.*.new_state_base.?[0] = 0;
    dtable.*.accuracy_log = 0;
}

/// Deep copy a decoding table, so that it can be used and free'd without
/// impacting the source table.
fn FSE_copy_dtable(dst: *FSETable, src: *const FSETable) void {
    print(@src().fn_name ++ ":\n", .{});
    if (src.*.accuracy_log == 0) {
        //memset(dst, 0, sizeof(FSETable));
        dst.* = src.*;
        return;
    }

    @breakpoint();
    //     size_t size = (size_t)1 << src->accuracy_log;
    //     dst->accuracy_log = src->accuracy_log;

    //     dst->symbols = malloc(size);
    //     dst->num_bits = malloc(size);
    //     dst->new_state_base = malloc(size * sizeof(u16));
    //     if (!dst->symbols || !dst->num_bits || !dst->new_state_base) {
    //         BAD_ALLOC();
    //     }

    //     memcpy(dst->symbols, src->symbols, size);
    //     memcpy(dst->num_bits, src->num_bits, size);
    //     memcpy(dst->new_state_base, src->new_state_base, size * sizeof(u16));
}

//******* ZSTD HELPER STRUCTS AND PROTOTYPES

/// A small structure that can be reused in various places that need to access
/// frame header information
const ZStdFrameHeader = struct {
    const Self = @This();

    /// The size of window that we need to be able to contiguously store for
    /// references
    window_size: usize = 0,
    /// The total output size of this compressed frame
    frame_content_size: usize = 0,

    /// The dictionary id if this frame uses one
    dictionary_id: u32 = 0,

    /// Whether or not the content of this frame has a checksum
    content_checksum_flag: bool = false,
    /// Whether or not the output for this frame is in a single segment
    single_segment_flag: bool = false,

    fn clear(s: *Self) void {
        s.window_size = 0;
        s.frame_content_size = 0;
        s.dictionary_id = 0;
        s.content_checksum_flag = false;
        s.single_segment_flag = false;
    }
};

/// The context needed to decode blocks in a frame
const ZStdFrameContext = struct {
    const Self = @This();

    allocator: *Allocator,
    header: ZStdFrameHeader = ZStdFrameHeader {},

    /// The total amount of data available for backreferences, to determine if an
    /// offset too large to be correct
    current_total_output: usize = 0,

    //const u8 *dict_content;
    //size_t dict_content_len;
    dict_content: ?[]u8 = null,
    dict_content_len: usize = 0,

    /// Entropy encoding tables so they can be repeated by future blocks instead
    /// of retransmitting
    literals_dtable: HUFTable = HUFTable {},
    ll_dtable: FSETable = FSETable {},
    ml_dtable: FSETable = FSETable {},
    of_dtable: FSETable = FSETable {},

    // The last 3 offsets for the special "repeat offsets".
    previous_offsets: [3]u64 = [_]u64{0} ** 3,

    /// A dictionary acts as initializing values for the frame context before
    /// decompression, so we implement it by applying it's predetermined
    /// tables and content to the context before beginning decompression
    fn apply_dict(s: *Self, dict: *const ZStdDictionary) !void {
        // If the content pointer is NULL then it must be an empty dict
        if (dict.*.content == null) {
            return;
        }

        // If the requested dictionary_id is non-zero, the correct
        // dictionary must be present
        if ((s.*.header.dictionary_id != 0) and (s.*.header.dictionary_id != dict.*.dictionary_id)) {
            return error.WrongDictionaryProvided;
        }

        // Copy the dict content to the context for references during sequence
        // execution
        s.*.dict_content = dict.*.content;
        s.*.dict_content_len = dict.*.content_size;

        // If it's a formatted dict copy the precomputed tables in so they can
        // be used in the table repeat modes
        if (dict.*.dictionary_id != 0) {
            // TODO
            @breakpoint();
            // Deep copy the entropy tables so they can be freed independently of
            // the dictionary struct
            // HUF_copy_dtable(&ctx->literals_dtable, &dict->literals_dtable);
            // FSE_copy_dtable(&ctx->ll_dtable, &dict->ll_dtable);
            // FSE_copy_dtable(&ctx->of_dtable, &dict->of_dtable);
            // FSE_copy_dtable(&ctx->ml_dtable, &dict->ml_dtable);

            // // Copy the repeated offsets
            // memcpy(ctx->previous_offsets, dict->previous_offsets,
            //        sizeof(ctx->previous_offsets));
        }
    }

    /// Decode data in a compressed block
    fn decompress_block(s: *Self, out: *ZStdOStream, in: *ZStdIStream) !void {
        // "A compressed block consists of 2 sections :
        //
        // Literals_Section
        // Sequences_Section"

        // Part 1: decode the literals block
        const literals = try s.decode_literals(in);
        const lmax = @minimum(literals.len, 64) - 1;
        print("---- Got {} literals '{s}'\n", .{ literals.len, literals[0..lmax] });

        // Part 2: decode the sequences block
        const sequences = try decode_sequences(s, in);

        // Part 3: combine literals and sequence commands to generate output
        if (sequences) |seqs| {
            try execute_sequences(s, out, literals, seqs);
        } else {
            // Copy literals to output?!?
            var uto = try out.get_write_ptr(literals.len);
            for (literals) |e, i| {
                uto[i] = e;
            }
        }

        // Free the resources
        s.*.allocator.free(literals);
        if (sequences) |seqs| {
            s.*.allocator.free(seqs);
        }
    }


    /// Free/clear the context
    fn free_context(s: *Self) void {
        // dict_content?
        s.*.literals_dtable.free_dtable(s.*.allocator);

        s.*.ll_dtable.free_dtable(s.*.allocator);
        s.*.ml_dtable.free_dtable(s.*.allocator);
        s.*.of_dtable.free_dtable(s.*.allocator);

        s.header.clear();
        // memset(context, 0, sizeof(frame_context_t));
    }

    /// Decode the literals section of a block
    fn decode_literals(s: *Self, in: *ZStdIStream) ![]u8 {
        // "Literals can be stored uncompressed or compressed using Huffman prefix
        // codes. When compressed, an optional tree description can be present,
        // followed by 1 or 4 streams."
        //
        // "Literals_Section_Header
        //
        // Header is in charge of describing how literals are packed. It's a
        // byte-aligned variable-size bitfield, ranging from 1 to 5 bytes, using
        // little-endian convention."
        //
        // "Literals_Block_Type
        //
        // This field uses 2 lowest bits of first byte, describing 4 different block
        // types"
        //
        // size_format takes between 1 and 2 bits
        const lblock_type = try in.*.read_bits(2);
        const lsize_format = try in.*.read_bits(2);

        print(@src().fn_name ++ ": lblock_type={}, lsize_format={}\n",
              .{ lblock_type, lsize_format });

        if (lblock_type <= 1) {
            // Raw or RLE literals block
            return try s.decode_literals_simple(in, lblock_type, lsize_format);
        } else {
            // Huffman compressed literals
            return s.decode_literals_compressed(in, lblock_type, lsize_format);
        }
    }

    /// Decodes literals blocks in raw or RLE form
    inline fn decode_literals_simple(s: *Self, in: *ZStdIStream,
                                     block_type: usize, size_format: usize) ![]u8 {
        var size: usize = 0;
        switch (size_format) {
            // These cases are in the form ?0
            // In this case, the ? bit is actually part of the size field
            0, 2 => {
                // "Size_Format uses 1 bit. Regenerated_Size uses 5 bits (0-31)."
                try in.*.rewind_bits(1);
                size = try in.*.read_bits(5);
            },
            1 => {
                // "Size_Format uses 2 bits. Regenerated_Size uses 12 bits (0-4095)."
                size = try in.*.read_bits(12);
            },
            3 => {
                // "Size_Format uses 2 bits. Regenerated_Size uses 20 bits (0-1048575)."
                size = try in.*.read_bits(20);
            },
            else => unreachable,
        }

        print(@src().fn_name ++ ": size={}\n", .{ size });
        if (size > MAX_LITERALS_SIZE) {
            return error.TooManyLiterals;
        }

        var literals = try s.*.allocator.alloc(u8, size);
        print(@src().fn_name ++ ": literal.len={}\n", .{ literals.len });
        // if (!*literals) {
        //     BAD_ALLOC();
        // }
        
        switch (block_type) {
            0 => {
                // "Raw_Literals_Block - Literals are stored uncompressed."
                print(@src().fn_name ++ ":*** Raw_Literals_Block\n", .{});
                const slice = try in.*.get_read_ptr(size);
                print("slice={s}\n", .{ slice, });
                // memcpy(*literals, read_ptr, size);
                for (slice) |e, i| {
                    literals[i] = e;
                }
            },
            1 => {
                // "RLE_Literals_Block - Literals consist of a single byte value repeated N times."
                print(@src().fn_name ++ ":RLE_Literals_Block\n", .{});
                // const u8 *const read_ptr = IO_get_read_ptr(in, 1);
                // memset(*literals, read_ptr[0], size);
                const slice = try in.*.get_read_ptr(1);
                for (literals) |*e| {
                    e.* = slice[0];
                }
            },
            else => unreachable,
        }

        return literals;
    }


    /// Decodes Huffman compressed literals
    fn decode_literals_compressed(s: *Self,
                                  in: *ZStdIStream,
                                  block_type: u64,
                                  size_format: u64) ![]u8 {
        var regenerated_size: usize = 0;
        var compressed_size: usize = 0;
        // Only size_format=0 has 1 stream, so default to 4
        var num_streams: usize = 4;

        print(@src().fn_name ++ ": block_type={}, size_format={}\n", .{
            block_type, size_format,
        });
        switch (size_format) {
            0 =>  {
                // "A single stream. Both Compressed_Size and Regenerated_Size use 10
                // bits (0-1023)."
                num_streams = 1;
                regenerated_size = try in.*.read_bits(10);
                compressed_size = try in.*.read_bits(10);
            },
            1 => {
                // "4 streams. Both Compressed_Size and Regenerated_Size use 10 bits
                // (0-1023)."
                regenerated_size = try in.*.read_bits(10);
                compressed_size = try in.*.read_bits(10);
            },
            2 => {
                // "4 streams. Both Compressed_Size and Regenerated_Size use 14 bits
                // (0-16383)."
                regenerated_size = try in.*.read_bits(14);
                compressed_size = try in.*.read_bits(14);
            },
            3 => {
                // "4 streams. Both Compressed_Size and Regenerated_Size use 18 bits
                // (0-262143)."
                regenerated_size = try in.*.read_bits(18);
                compressed_size = try in.*.read_bits(18);
            },
            else => unreachable,
        }

        if (regenerated_size > MAX_LITERALS_SIZE) {
            print(@src().fn_name ++ ": {} regenerated_size={}, MAX_LITERALS_SIZE={}\n",
                  .{ size_format, regenerated_size, MAX_LITERALS_SIZE, });
            return error.Corruption;
        }

        if (debug) {
            print(@src().fn_name ++ ": {} regenerated_size={}, compressed_size={}\n",
                  .{ size_format, regenerated_size, compressed_size, });
        }
        var literals = try s.*.allocator.alloc(u8, regenerated_size);
        // if (!*literals) {
        //     BAD_ALLOC();
        // }
        print(@src().fn_name ++ ": literal.len={}\n", .{ literals.len });

        var lit_stream = ZStdOStream.from_slice(literals[0..]);
        lit_stream.magic = 0x12345678;
        var huf_stream = try in.*.make_sub_istream(compressed_size);

        if (block_type == 2) {
            // Decode the provided Huffman table
            // "This section is only present when Literals_Block_Type type is
            // Compressed_Literals_Block (2)."

            s.*.literals_dtable.free_dtable(s.*.allocator);
            try decode_huf_table(s.allocator, &s.*.literals_dtable, &huf_stream);
        } else {
            // If the previous Huffman table is being repeated, ensure it exists
            if (s.*.literals_dtable.symbols) |_| {
            } else {
                // CORRUPTION();
                print(@src().fn_name ++
                          "If the previous Huffman table is being repeated, ensure it exists\n", .{});
                return error.Corruption;
            }
            
        }

        //var symbols_decoded: usize = 0;
        print(@src().fn_name ++ ": num_streams={}\n", .{ num_streams });
        const symbols_decoded = if (num_streams == 1)
            try s.*.literals_dtable.decompress_1stream(&lit_stream, &huf_stream)
            else
            try s.*.literals_dtable.decompress_4stream(&lit_stream, &huf_stream);

        print(@src().fn_name ++ ":### symbols_decoded={}, regenerated_size={}\n",
              .{ symbols_decoded, regenerated_size, });
        if (symbols_decoded != regenerated_size) {
            return error.Corruption;
        }

        return literals;
    }
};


/// A tuple containing the parts necessary to decode and execute a ZSTD sequence
/// command
const ZStdSequenceCommand = struct {
    literal_length: u32 = 0,
    match_length: u32 = 0,
    offset: u32 = 0,
};

// The decoder works top-down, starting at the high level like Zstd frames, and
// working down to lower more technical levels such as blocks, literals, and
// sequences.  The high-level functions roughly follow the outline of the
// format specification:
// https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md

// Decode a single Zstd frame, or error if the input is not a valid frame.
// Accepts a dict argument, which may be NULL indicating no dictionary.
// See
// https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#frame-concatenation

//******* END ZSTD HELPER STRUCTS AND PROTOTYPES

const ZStdDecoderContext = struct {
    const Self = @This();

    in: ZStdIStream,
    out: ZStdOStream,

    // pointer or not?!?
    allocator: *Allocator,

    fn from_slices(in: []const u8, out: []u8, allocator: *Allocator) Self {
        return Self {
            .in = ZStdIStream.from_slice(in),
            .out = ZStdOStream.from_slice(out),
            .allocator = allocator,
        };
    }

    /// Decode a single Zstd frame, or error if the input is not a valid frame.
    /// Accepts a dict argument, which may be NULL indicating no dictionary.
    /// See
    /// https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#frame-concatenation
    fn decode_frame(s: *Self, dict: *const ZStdDictionary) !void {
        const magic_number: u32 = @truncate(u32, s.*.in.read_bits(32) catch 0);
        // Zstandard frame
        //
        // "Magic_Number
        //
        // 4 Bytes, little-endian format. Value : 0xFD2FB528"
        if (magic_number != ZSTD_MAGIC) {
            // not a real frame or skippable frame
            // ERROR("Tried to decode non-ZSTD frame");
            print("frame magic=0x{x}\n", .{ magic_number });
            return error.InvalidMagic;
        }

        // ZSTD frame
        try s.*.decode_data_frame(dict);
        
        return;
    }

    fn parse_frame_header(s: *Self) !ZStdFrameHeader {
        // "The first header's byte is called the Frame_Header_Descriptor. It tells
        // which other fields are present. Decoding this byte is enough to tell the
        // size of Frame_Header.
        //
        // Bit number   Field name
        // 7-6  Frame_Content_Size_flag
        // 5    Single_Segment_flag
        // 4    Unused_bit
        // 3    Reserved_bit
        // 2    Content_Checksum_flag
        // 1-0  Dictionary_ID_flag"
        const descriptor: u8 = @truncate(u8, try s.*.in.read_bits(8));

        // decode frame header descriptor into flags
        const reserved_bit = (descriptor >> 3) & 1;

        if (reserved_bit != 0) {
            // CORRUPTION();
            return error.InvalidFrameHeaderReservedBit;
        }

        const frame_content_size_flag = descriptor >> 6;
        const dictionary_id_flag = descriptor & 3;

        var frame_header = ZStdFrameHeader {};
        frame_header.single_segment_flag = ((descriptor >> 5) & 1) == 1;
        frame_header.content_checksum_flag = ((descriptor >> 2) & 1) == 1;

        // decode window size
        if (!frame_header.single_segment_flag) {
            // "Provides guarantees on maximum back-reference distance that will be
            // used within compressed data. This information is important for
            // decoders to allocate enough memory.
            //
            // Bit numbers  7-3         2-0
            // Field name   Exponent    Mantissa"
            const window_descriptor: u8 = @truncate(u8, try s.*.in.read_bits(8));
            const exponent = window_descriptor >> 3;
            const mantissa = window_descriptor & 7;

            // Use the algorithm from the specification to compute
            // window size
            // https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#window_descriptor
            const window_base: usize = @as(usize, 1) << @truncate(u6, 10 + exponent);
            const window_add: usize = (window_base / 8) * mantissa;
            frame_header.window_size = window_base + window_add;
        }

        // decode dictionary id if it exists
        if (dictionary_id_flag != 0) {
            // "This is a variable size field, which contains the ID of the
            // dictionary required to properly decode the frame. Note that this
            // field is optional. When it's not present, it's up to the caller to
            // make sure it uses the correct dictionary. Format is little-endian."
            const bytes_array: [4]u32 = [_]u32{0, 1, 2, 4};
            const bytes = bytes_array[dictionary_id_flag];

            frame_header.dictionary_id = @truncate(u32, try s.*.in.read_bits(bytes * 8));
        } else {
            frame_header.dictionary_id = 0;
        }

        // decode frame content size if it exists
        if (frame_header.single_segment_flag or frame_content_size_flag > 0) {
            // "This is the original (uncompressed) size. This information is
            // optional. The Field_Size is provided according to value of
            // Frame_Content_Size_flag. The Field_Size can be equal to 0 (not
            // present), 1, 2, 4 or 8 bytes. Format is little-endian."
            //
            // if frame_content_size_flag == 0 but single_segment_flag is set, we
            // still have a 1 byte field
            const bytes_array: [4]u32 = [_]u32{1, 2, 4, 8};
            const bytes = bytes_array[frame_content_size_flag];

            frame_header.frame_content_size = try s.*.in.read_bits(bytes * 8);
            if (bytes == 2) {
                // "When Field_Size is 2, the offset of 256 is added."
                frame_header.frame_content_size += 256;
            }
        } else {
            frame_header.frame_content_size = 0;
        }

        if (frame_header.single_segment_flag) {
            // "The Window_Descriptor byte is optional. It is absent when
            // Single_Segment_flag is set. In this case, the maximum back-reference
            // distance is the content size itself, which can be any value from 1 to
            // 2^64-1 bytes (16 EB)."
            frame_header.window_size = frame_header.frame_content_size;
        }

        print("frame_header={}\n", .{ frame_header });
        return frame_header;
    }

    /// Takes the information provided in the header and dictionary, and initializes
    /// the context for this frame
    fn init_frame_context(s: *Self, frame_header: *const ZStdFrameHeader,
                          dict: *const ZStdDictionary) !ZStdFrameContext {
        // static void init_frame_context(frame_context_t *const context,
        //                                istream_t *const in,
        //                                const dictionary_t *const dict) {
        // Most fields in context are correct when initialized to 0
        // memset(context, 0, sizeof(frame_context_t));

        // NOTE: We already did this: Parse data from the frame header
        // parse_frame_header(&context->header, in);

        // Set up the offset history for the repeat offset commands
        //     context->previous_offsets[0] = 1;
        //     context->previous_offsets[1] = 4;
        //     context->previous_offsets[2] = 8;
        _ = s;

        var fctx = ZStdFrameContext { .allocator = s.*.allocator,
                                     .header = frame_header.*,
                                     .previous_offsets = [_]u64{1, 4, 8},
                                     };
        // Apply details from the dict if it exists
        try fctx.apply_dict(dict);

        return fctx;
    }


    /// Decode a frame that contains compressed data.  Not all frames
    /// do as there are skippable frames.  See
    /// https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#general-structure-of-zstandard-frame-format
    fn decode_data_frame(s: *Self, dict: *const ZStdDictionary) !void {
        // fn decode_data_frame(ostream_t *const out, istream_t *const in,
        //                               const dictionary_t *const dict) !void {
        const frame_header = try s.*.parse_frame_header();

        // Initialize the context that needs to be carried from block to block
        var fctx = try s.*.init_frame_context(&frame_header, dict);

        if (fctx.header.frame_content_size != 0 and
                (fctx.header.frame_content_size > s.*.out.ptr.len)) {
            // OUT_SIZE();
            return error.OutputBufferTooSmall;
        }

        try s.*.decompress_data(&fctx);

        fctx.free_context();
    }

    /// Decompress the data from a frame block by block
    fn decompress_data(s: *Self, fctx: *ZStdFrameContext) !void {
        // static void decompress_data(frame_context_t *const ctx, ostream_t *const out,
        //                             istream_t *const in) {
        // "A frame encapsulates one or multiple blocks. Each block can be
        // compressed or not, and has a guaranteed maximum content size, which
        // depends on frame parameters. Unlike frames, each block depends on
        // previous blocks for proper decoding. However, each block can be
        // decompressed without waiting for its successor, allowing streaming
        // operations."
        //     int last_block = 0;
        while (true) {
            // do {
            // "Last_Block
            //
            // The lowest bit signals if this block is the last one. Frame ends
            // right after this block.
            //
            // Block_Type and Block_Size
            //
            // The next 2 bits represent the Block_Type, while the remaining 21 bits
            // represent the Block_Size. Format is little-endian."
            const last_block = (try s.*.in.read_bits(1)) == 1;
            const block_type = try s.*.in.read_bits(2);
            const block_len: usize = @as(usize, try s.*.in.read_bits(21));

            print("decompress_data: last_block={}, block_type={}, block_len={}\n", .{
                last_block, block_type, block_len,
            });

            switch (block_type) {
                0 => {
                    // "Raw_Block - this is an uncompressed block. Block_Size is the
                    // number of bytes to read and copy."
                    // const u8 *const read_ptr = IO_get_read_ptr(in, block_len);
                    // u8 *const write_ptr = IO_get_write_ptr(out, block_len);

                    var ufrom = try s.*.in.get_read_ptr(block_len);
                    var uto = try s.*.out.get_write_ptr(block_len);

                    // Copy the raw data into the output
                    // memcpy(write_ptr, read_ptr, block_len);
                    for (ufrom) |e, i| {
                        uto[i] = e;
                    }

                    fctx.*.current_total_output += block_len;
                },
                1 => {
                    // "RLE_Block - this is a single byte, repeated N times. In which
                    // case, Block_Size is the size to regenerate, while the
                    // "compressed" block is just 1 byte (the byte to repeat)."

                    // const u8 *const read_ptr = IO_get_read_ptr(in, 1);
                    // u8 *const write_ptr = IO_get_write_ptr(out, block_len);
                    // Copy `block_len` copies of `read_ptr[0]` to the output
                    // memset(write_ptr, read_ptr[0], block_len);
                    // ctx->current_total_output += block_len;
                    // break;
                    print("RLE block: size={}\n", .{ block_len });
                    var src = try s.*.in.get_read_ptr(1);
                    var dst = try s.*.out.get_write_ptr(block_len);
                    {
                        const data = src[0];
                        var i: usize = 0;
                        while (i < block_len) : (i += 1) {
                            dst[i] = data;
                        }
                    }

                    fctx.*.current_total_output += block_len;
                },
                2 => {
                    // "Compressed_Block - this is a Zstandard compressed block,
                    // detailed in another section of this specification. Block_Size is
                    // the compressed size.
                    print("Compressed block: size={}\n", .{ block_len });

                    // Create a sub-stream for the block
                    var block_stream = try s.*.in.make_sub_istream(block_len);
                    try fctx.*.decompress_block(&s.*.out, &block_stream);
                },
                3 => {
                    // "Reserved - this is not a block. This value cannot be used with
                    // current version of this specification."
                    // CORRUPTION();
                    return error.InvalidBlockType;
                },
                // IMPOSSIBLE();
                else => unreachable,
            }
            // } while (!last_block);
            if (last_block) {
                break;
            }
        }
        if (fctx.*.header.content_checksum_flag) {
            // This program does not support checking the checksum, so skip over it
            // if it's present
            // IO_advance_input(in, 4);
            // xxh64 lower 4 bytes, little endian
            const ck = try s.*.in.get_read_ptr(4);
            const cksum: u32 = (@as(u32, ck[3]) << 24) | (@as(u32, ck[2]) << 16)
                | (@as(u32, ck[1]) << 8) | (@as(u32, ck[0]));

            print("*** Skipping 32 bit checksum: {x:08} ***\n", .{ cksum } );
        }
    }
};



// ======================================================================

pub fn ZSTD_decompress(dst: []u8, src: []const u8, allocator: *Allocator) !usize {
    if (dst.len == 0) {
        return error.OutputBufferTooSmall;
    }
    //dictionary_t* uninit_dict = create_dictionary();
    var uninit_dict = ZStdDictionary.new_uninit();
    print("uninit_dict={}\n", .{ uninit_dict });
    const decomp_size = ZSTD_decompress_with_dict(dst, src, &uninit_dict, allocator);
    //free_dictionary(uninit_dict);
    return decomp_size;
}

pub fn ZSTD_decompress_with_dict(dst: []u8, src: []const u8, parsed_dict: *ZStdDictionary,
                             allocator: *Allocator) !usize {

    var zdctx = ZStdDecoderContext.from_slices(src, dst, allocator);
    print("zdctx={}\n", .{ zdctx });

    // "A content compressed by Zstandard is transformed into a Zstandard frame.
    // Multiple frames can be appended into a single file or stream. A frame is
    // totally independent, has a defined beginning and end, and a set of
    // parameters which tells the decoder how to decompress it."

    // this decoder assumes decompression of a single frame
    try zdctx.decode_frame(parsed_dict);

    return zdctx.out.pos;
}


/// Decode the Huffman table description
fn decode_huf_table(allocator: *Allocator, dtable: *HUFTable, in: *ZStdIStream) !void {
    // "All literal values from zero (included) to last present one (excluded)
    // are represented by Weight with values from 0 to Max_Number_of_Bits."

    // "This is a single byte value (0-255), which describes how to decode the list of weights."
    const header = @truncate(u8, try in.read_bits(8));

    print("=> decode_huf_table: header={}\n", .{ header });
    var weights: [HUF_MAX_SYMBS]u8 = [_]u8{0} ** HUF_MAX_SYMBS;

    var num_symbs: i64 = 0;

    if (header >= 128) {
        // "This is a direct representation, where each Weight is written
        // directly as a 4 bits field (0-15). The full representation occupies
        // ((Number_of_Symbols+1)/2) bytes, meaning it uses a last full byte
        // even if Number_of_Symbols is odd. Number_of_Symbols = headerByte -
        // 127"
        num_symbs = header - 127;

        const bytes = (@bitCast(usize, num_symbs) + 1) / 2;
        const weight_src = try in.*.get_read_ptr(bytes);
        // "They are encoded forward, 2
        // weights to a byte with the first weight taking the top four bits
        // and the second taking the bottom four (e.g. the following
        // operations could be used to read the weights: Weight[0] =
        // (Byte[0] >> 4), Weight[1] = (Byte[0] & 0xf), etc.)."
        for (weight_src) |e, i| {
            const idx = i * 2;
            weights[idx + 0] = (e >> 4) & 0xf;
            weights[idx + 1] =  e       & 0xf;
        }
    } else {
        // The weights are FSE encoded, decode them before we can construct the
        // table
        var fse_stream = try in.*.make_sub_istream(header);
        var weight_stream = ZStdOStream.from_slice(weights[0..]);
        weight_stream.magic = 0x87654321;
        try fse_decode_hufweights(allocator, &weight_stream, &fse_stream, &num_symbs);
    }

    // Construct the table using the decoded weights
    try dtable.init_usingweights(weights[0..], num_symbs, allocator);
}

fn fse_decode_hufweights(allocator: *Allocator, weights: *ZStdOStream, in: *ZStdIStream,
                         num_symbs: *i64) !void {
    const MAX_ACCURACY_LOG = 7;

    var dtable: FSETable = FSETable {};

    // "An FSE bitstream starts by a header, describing probabilities
    // distribution. It will create a Decoding Table. For a list of Huffman
    // weights, maximum accuracy is 7 bits."
    try FSE_decode_header(&dtable, in, MAX_ACCURACY_LOG, allocator);

    // Decode the weights
    num_symbs.* = try FSE_decompress_interleaved2(&dtable, weights, in);

    dtable.free_dtable(allocator);
}
// //******* END LITERALS DECODING

// //******* SEQUENCE DECODING
// /// The combination of FSE states needed to decode sequences
const ZStdSequenceStates = struct {
    const Self = @This();

    ll_state: u16 = 0,
    ll_table: FSETable = FSETable {},
    of_state: u16 = 0,
    of_table: FSETable = FSETable {},
    ml_state: u16 = 0,
    ml_table: FSETable = FSETable {},


    inline fn peek_current_ll_symbol(s: *const Self) !u8 {
        return s.*.ll_table.peek_symbol(s.*.ll_state);
    }

    inline fn peek_current_of_symbol(s: *const Self) !u8 {
        return s.*.of_table.peek_symbol(s.*.of_state);
    }

    inline fn peek_current_ml_symbol(s: *const Self) !u8 {
        return s.*.ml_table.peek_symbol(s.*.ml_state);
    }
};

/// Different modes to signal to decode_seq_tables what to do
const ZStdSequencePart = enum(u8) {
    LiteralLength = 0,
    Offset = 1,
    MatchLength = 2,
};

const ZStdSequenceMode = enum(u8) {
    Predefined = 0,
    RLE = 1,
    FSE = 2,
    Repeat = 3,
};

/// The predefined FSE distribution tables for `Predefined` mode
const SEQ_LITERAL_LENGTH_DEFAULT_DIST: [36]i16 = [_]i16 {
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,  1,  2,  2,
    2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1
};

const SEQ_OFFSET_DEFAULT_DIST: [29]i16 = [_]i16 {
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1,  1,  1,  1,  1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1
};
const SEQ_MATCH_LENGTH_DEFAULT_DIST: [53]i16 = [_]i16 {
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1,  1,  1,  1,  1,  1,  1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1
};

// // The sequence decoding baseline and number of additional bits to
// // read/add
// // https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#the-codes-for-literals-lengths-match-lengths-and-offsets
const SequenceInfo = struct {
    const Self = @This();

    baseline: u32,
    extrabits: u8
};

const SequenceLiterals: [36]SequenceInfo = [_]SequenceInfo {
    
};

const SEQ_LITERAL_LENGTH_BASELINES: [36]u32 = [_]u32 {
    0,  1,  2,   3,   4,   5,    6,    7,    8,    9,     10,    11,
    12, 13, 14,  15,  16,  18,   20,   22,   24,   28,    32,    40,
    48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
};
const SEQ_LITERAL_LENGTH_EXTRA_BITS: [36]u8 = [_]u8 {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  1,
    1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
};

const SEQ_MATCH_LENGTH_BASELINES: [53]u32 = [_]u32 {
    3,  4,   5,   6,   7,    8,    9,    10,   11,    12,    13,   14, 15, 16,
    17, 18,  19,  20,  21,   22,   23,   24,   25,    26,    27,   28, 29, 30,
    31, 32,  33,  34,  35,   37,   39,   41,   43,    47,    51,   59, 67, 83,
    99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539
};
const SEQ_MATCH_LENGTH_EXTRA_BITS: [53]u8 = [_]u8 {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  1,  1, 1,
    2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
};

/// Offset decoding is simpler so we just need a maximum code value
const SEQ_MAX_CODES: [3]u8 = [_]u8{35, 255, 52};

/// Decode the sequences part of a block
fn decode_sequences(fctx: *ZStdFrameContext, in: *ZStdIStream) !?[]ZStdSequenceCommand {
    // "A compressed block is a succession of sequences . A sequence is a
    // literal copy command, followed by a match copy command. A literal copy
    // command specifies a length. It is the number of bytes to be copied (or
    // extracted) from the literal section. A match copy command specifies an
    // offset and a length. The offset gives the position to copy from, which
    // can be within a previous block."

    var num_sequences: usize = 0;

    // "Number_of_Sequences
    //
    // This is a variable size field using between 1 and 3 bytes. Let's call its
    // first byte byte0."
    const header = try in.*.read_bits(8);
    print("decode_sequences: header={x}\n", .{ header });
    if (header == 0) {
        // "There are no sequences. The sequence section stops there.
        // Regenerated content is defined entirely by literals section."
        //         *sequences = NULL;
        //         return 0;
        return null;
    } else if (header < 128) {
        // "Number_of_Sequences = byte0 . Uses 1 byte."
        num_sequences = header;
    } else if (header < 255) {
        // "Number_of_Sequences = ((byte0-128) << 8) + byte1 . Uses 2 bytes."
        const extra = try in.*.read_bits(8);
        num_sequences = ((header - 128) << 8) + extra;
    } else {
        // "Number_of_Sequences = byte1 + (byte2<<8) + 0x7F00 . Uses 3 bytes."
        num_sequences = (try in.*.read_bits(16)) + 0x7F00;
    }
    print("num_sequences={}\n", .{ num_sequences });
    var sequences = try fctx.*.allocator.alloc(ZStdSequenceCommand, num_sequences);
    //     if (!*sequences) {
    //         BAD_ALLOC();
    //     }

    try decompress_sequences(fctx, in, sequences[0..]);
    return sequences;
}

/// Decompress the FSE encoded sequence commands
fn decompress_sequences(fctx: *ZStdFrameContext, in: *ZStdIStream, sequences: []ZStdSequenceCommand) !void {
    // "The Sequences_Section regroup all symbols required to decode commands.
    // There are 3 symbol types : literals lengths, offsets and match lengths.
    // They are encoded together, interleaved, in a single bitstream."

    // "Symbol compression modes
    //
    // This is a single byte, defining the compression mode of each symbol
    // type."
    //
    // Bit number : Field name
    // 7-6        : Literals_Lengths_Mode
    // 5-4        : Offsets_Mode
    // 3-2        : Match_Lengths_Mode
    // 1-0        : Reserved
    var compression_modes = try in.*.read_bits(8);
    if (debug) {
        print("decompress_sequences: compression_modes={x}, sequences.len={}\n", .{
            compression_modes, sequences.len,
        });
    }

    if ((compression_modes & 3) != 0) {
        // Reserved bits set
        return error.IncalidCompressModeReservedBits;
    }

    // "Following the header, up to 3 distribution tables can be described. When
    // present, they are in this order :
    //
    // Literals lengths
    // Offsets
    // Match Lengths"
    // Update the tables we have stored in the context
    const literal_mode = @intToEnum(ZStdSequenceMode, @truncate(u2, (compression_modes >> 6) & 3));
    const offset_mode = @intToEnum(ZStdSequenceMode, @truncate(u2, (compression_modes >> 4) & 3));
    const match_mode = @intToEnum(ZStdSequenceMode, @truncate(u2, (compression_modes >> 2) & 3));
    print("=> decompress_sequences: literal_mode={}, offset_mode={}, match_mode={}\n",
          .{ literal_mode, offset_mode, match_mode, });
    // HACK, check this!!!
    try decode_seq_table(&fctx.*.ll_dtable, in, .LiteralLength, literal_mode, fctx.*.allocator);
    try decode_seq_table(&fctx.*.of_dtable, in, .Offset, offset_mode, fctx.*.allocator);
    try decode_seq_table(&fctx.*.ml_dtable, in, .MatchLength, match_mode, fctx.*.allocator);
    print("fctx.*. ll_dtable={}, of_dtable={}, ml_dtable={}\n", .{
        fctx.*.ll_dtable, fctx.*.of_dtable, fctx.*.ml_dtable,
    });


    var states = ZStdSequenceStates {
        .ll_table = fctx.*.ll_dtable,
        .of_table = fctx.*.of_dtable,
        .ml_table = fctx.*.ml_dtable,
    };

    const len = in.*.length();
    const src = try in.*.get_read_ptr(len);

    print("length is {}\n", .{ len });

    // "After writing the last bit containing information, the compressor writes
    // a single 1-bit and then fills the byte with 0-7 0 bits of padding."
    const padding = 8 - highest_set_bit(src[len - 1]);
    const upadding = @bitCast(u16, padding);
    // The offset starts at the end because FSE streams are read backwards
    var bit_offset = ((len * 8) - upadding);

    print("padding={}/{}, bit_offset={}\n", .{ padding, upadding, bit_offset, });
    // "The bitstream starts with initial state values, each using the required
    // number of bits in their respective accuracy, decoded previously from
    // their normalized distribution.
    //
    // It starts by Literals_Length_State, followed by Offset_State, and finally
    // Match_Length_State."
    var bo = @bitCast(i64, bit_offset);
    print("decompress_sequences: bit_offset={}\n", .{ bo });
    try states.ll_table.init_state(&states.ll_state, src, &bo);
    print("decompress_sequences: bit_offset={}\n", .{ bo });
    try states.of_table.init_state(&states.of_state, src, &bo);
    print("decompress_sequences: bit_offset={}\n", .{ bo });
    try states.ml_table.init_state(&states.ml_state, src, &bo);
    print("decompress_sequences: bit_offset={}\n", .{ bo });

    for (sequences) |*e| {
        e.* = try decode_sequence(&states, src, &bo);
    }

    if (bo != 0) {
        print("CORRUPTION bo={}\n", .{ bo });
        return error.Corruption;
    }
}

/// Decode a single sequence and update the state
fn decode_sequence(states: *ZStdSequenceStates, src: []const u8,
                   offset: *i64) !ZStdSequenceCommand {
    // "Each symbol is a code in its own context, which specifies Baseline and
    // Number_of_Bits to add. Codes are FSE compressed, and interleaved with raw
    // additional bits in the same bitstream."

    // Decode symbols, but don't update states
    const of_code = try states.peek_current_of_symbol();
    const ll_code = try states.peek_current_ll_symbol();
    const ml_code = try states.peek_current_ml_symbol();

    // print("decode_sequence: codes of={}, ll={}, ml={}, offset={}\n", .{
    //     of_code, ll_code, ml_code, offset.*,
    // });

    // Offset doesn't need a max value as it's not decoded using a table
    if (ll_code > SEQ_MAX_CODES[@enumToInt(ZStdSequencePart.LiteralLength)]
            or ml_code > SEQ_MAX_CODES[@enumToInt(ZStdSequencePart.MatchLength)]) {
        // CORRUPTION();
        print("decode_sequence: Corruption\n", .{});
        return error.Corruption;
    }

    // Read the interleaved bits
    // "Decoding starts by reading the Number_of_Bits required to decode Offset.
    // It then does the same for Match_Length, and then for Literals_Length."
    const of_bits = @truncate(u32, try STREAM_read_bits(src, of_code, offset));
    const ml_bits = @truncate(u32, try STREAM_read_bits(src, SEQ_MATCH_LENGTH_EXTRA_BITS[ml_code], offset));
    const ll_bits = @truncate(u32, try STREAM_read_bits(src, SEQ_LITERAL_LENGTH_EXTRA_BITS[ll_code], offset));

    const seq = ZStdSequenceCommand {.offset = (@as(u32, 1) << @truncate(u5, of_code)) + of_bits,
                                     .match_length = SEQ_MATCH_LENGTH_BASELINES[ml_code] + ml_bits,
                                     .literal_length = SEQ_LITERAL_LENGTH_BASELINES[ll_code] + ll_bits };
    // print("ll decode_sequence: seq={}, offset={}\n", .{ seq, offset.* });

    // "If it is not the last sequence in the block, the next operation is to
    // update states. Using the rules pre-calculated in the decoding tables,
    // Literals_Length_State is updated, followed by Match_Length_State, and
    // then Offset_State."
    // If the stream is complete don't read bits to update state
    if (offset.* != 0) {
        try states.*.ll_table.update_state(&states.*.ll_state, src, offset);
        try states.*.ml_table.update_state(&states.*.ml_state, src, offset);
        try states.*.of_table.update_state(&states.*.of_state, src, offset);
    }

    return seq;
}

/// Given a sequence part and table mode, decode the FSE distribution
/// Errors if the mode is `Repeat` without a pre-existing table in `table`
fn decode_seq_table(table: *FSETable, in: *ZStdIStream,
                    stype: ZStdSequencePart, mode: ZStdSequenceMode,
                    allocator: *Allocator) !void {
    print("decode_seq_table: stype={}, mode={}\n", .{ stype, mode, });
    // Constant arrays indexed by ZStdSequencePart
    const Defaults = struct {
        distribution: []const i16,
        length: usize,// this may be given by distribution.len
        accuracy: u8,
        max_accuracy: u6,
    };
    const default_values: [3]Defaults = [_]Defaults {
        Defaults {.distribution = SEQ_LITERAL_LENGTH_DEFAULT_DIST[0..],
                  .length = 36,
                  .accuracy = 6,
                  .max_accuracy = 9
                  },
        Defaults {.distribution = SEQ_OFFSET_DEFAULT_DIST[0..],
                  .length = 29,
                  .accuracy = 5,
                  .max_accuracy = 8},
        Defaults {.distribution = SEQ_MATCH_LENGTH_DEFAULT_DIST[0..],
                  .length = 53,
                  .accuracy = 6,
                  .max_accuracy = 9},
    };

    switch (mode) {
        .Predefined => {
            // "Predefined_Mode : uses a predefined distribution table."
            const eint = @enumToInt(stype);
            const defaults = default_values[eint];

            try FSE_init_dtable(table, defaults.distribution[0..],
                                defaults.length, defaults.accuracy, allocator);
        },
        .RLE => {
            // "RLE_Mode : it's a single code, repeated Number_of_Sequences times."
            // const u8 symb = IO_get_read_ptr(in, 1)[0];
            // FSE_init_dtable_rle(table, symb);
            const src = try in.*.get_read_ptr(1);
            try FSE_init_dtable_rle(table, src[0], allocator);
        },
        .FSE => {
            // "FSE_Compressed_Mode : standard FSE compression. A
            // distribution table will be present "
            const eint = @enumToInt(stype);
            const defaults = default_values[eint];
            try FSE_decode_header(table, in, defaults.max_accuracy, allocator);
        },
        .Repeat => {
            // "Repeat_Mode : re-use distribution table from previous compressed
            // block."
            // Nothing to do here, table will be unchanged
            if (table.*.symbols) |_| {
            } else {
                // This mode is invalid if we don't already have a table
                print("decode_seq_table: Corruption {}\n", .{ mode, });
                return error.Corruption;
            }
            // @breakpoint();
        },
        // TODO should be unreachable when this compiles cleanly
        // else => unreachable,
    }

}
// //******* END SEQUENCE DECODING

// //******* SEQUENCE EXECUTION
/// Execute the decoded sequences on the literals block
fn execute_sequences(fctx: *ZStdFrameContext, out: *ZStdOStream,
                     literals: []const u8,
                     sequences: []const ZStdSequenceCommand) !void {
    print("execute_sequences: literal_length={}, num_sequences={}\n", .{
        literals.len, sequences.len,
    });
    print("execute_sequences: previous_offsets={}/{}/{}, current_total_output={}\n", .{
        fctx.*.previous_offsets[0], fctx.*.previous_offsets[1],
        fctx.*.previous_offsets[2], fctx.*.current_total_output,
    });
    // TODO
    var offset_hist = fctx.*.previous_offsets;
    var litstream = ZStdIStream.from_slice(literals[0..]);

    var total_output = fctx.*.current_total_output;

    // for (size_t i = 0; i < num_sequences; i++) {
    //     const ZStdSequenceCommand seq = sequences[i];
    //     {
    //         const u32 literals_size = copy_literals(seq.literal_length, &litstream, out);
    //         total_output += literals_size;
    //     }
    for (sequences) |*e| {
        // print("#### e={}\n", .{ e.* });
        const literal_size = try copy_literals(e.*.literal_length, &litstream, out);
        total_output += literal_size;

        const offset = compute_offset(e, offset_hist[0..]);
        const match_length = e.*.match_length;
        //print("############## offset={}, match_length={}\n", .{ offset, match_length });

        try execute_match_copy(fctx, offset, match_length, total_output, out);

        total_output += match_length;
    }

    // Copy any leftover literals
    {
        const len = @truncate(u32, litstream.length());
        //print("execute_sequences: leftover literals={}, {}\n", .{ len, litstream.buf[0..] });
        _ = try copy_literals(len, &litstream, out);
        total_output += len;
    }

    fctx.*.current_total_output = total_output;
}

inline
fn copy_literals(literal_length: u32, litstream: *ZStdIStream,
                 out: *ZStdOStream) !u32 {

    // If the sequence asks for more literals than are left, the
    // sequence must be corrupted
    if (literal_length > litstream.*.length()) {
        print("copy_literals: literal_length={} vs {}\n", .{ literal_length, litstream.*.length() });
        return error.Corruption;
    }

    // Copy literals to output
    try out.write(try litstream.*.get_read_ptr(literal_length));

    return literal_length;
}

/// Given an offset code from a sequence command (either an actual offset value
/// or an index for previous offset), computes the correct offset and updates
/// the offset history
inline
fn compute_offset(seq: *const ZStdSequenceCommand, offset_hist: []u64) usize {
    var offset: usize = 0;
    // Offsets are special, we need to handle the repeat offsets
    if (seq.offset <= 3) {
        // "The first 3 values define a repeated offset and we will call
        // them Repeated_Offset1, Repeated_Offset2, and Repeated_Offset3.
        // They are sorted in recency order, with Repeated_Offset1 meaning
        // 'most recent one'".

        // Use 0 indexing for the array
        var idx = seq.offset - 1;
        if (seq.literal_length == 0) {
            // "There is an exception though, when current sequence's
            // literals length is 0. In this case, repeated offsets are
            // shifted by one, so Repeated_Offset1 becomes Repeated_Offset2,
            // Repeated_Offset2 becomes Repeated_Offset3, and
            // Repeated_Offset3 becomes Repeated_Offset1 - 1_byte."
            idx += 1;
        }

        if (idx == 0) {
            offset = offset_hist[0];
        } else {
            // If idx == 3 then literal length was 0 and the offset was 3,
            // as per the exception listed above
            offset = if (idx < 3) offset_hist[idx] else offset_hist[0] - 1;

            // If idx == 1 we don't need to modify offset_hist[2], since
            // we're using the second-most recent code
            if (idx > 1) {
                offset_hist[2] = offset_hist[1];
            }
            offset_hist[1] = offset_hist[0];
            offset_hist[0] = offset;
        }
    } else {
        // When it's not a repeat offset:
        // "if (Offset_Value > 3) offset = Offset_Value - 3;"
        offset = seq.offset - 3;

        // Shift back history
        offset_hist[2] = offset_hist[1];
        offset_hist[1] = offset_hist[0];
        offset_hist[0] = offset;
    }

    return offset;
}

/// Given an offset, match length, and total output, as well as the frame
/// context for the dictionary, determines if the dictionary is used and
/// executes the copy operation
fn execute_match_copy(fctx: *ZStdFrameContext, offset: usize,
                      match_length: usize, total_output: usize,
                      out: *ZStdOStream) !void {
    // print("execute_match_copy: total_output={}, window_size={}, offset={}, dcl={}\n", .{
    //     total_output, fctx.*.header.window_size, offset, fctx.*.dict_content_len,
    // });
    // const start = 0;
    // print("execute_match_copy: out.ptr={s}, pos={}, start={}\n", .{ out.*.ptr[start..out.*.pos], out.*.pos, start });
    //var write_slice = out.*;
    var lmatch_length = match_length;
    if (total_output <= fctx.*.header.window_size) {
        // In this case offset might go back into the dictionary
        if (offset > total_output + fctx.*.dict_content_len) {
            // The offset goes beyond even the dictionary
            print("execute_match_copy: Corruption\n", .{});
            return error.Corruption;
        }

        if (offset > total_output) {
            // "The rest of the dictionary is its content. The content act
            // as a "past" in front of data to compress or decompress, so it
            // can be referenced in sequence commands."
            const dict_copy = @minimum(offset - total_output, match_length);
            const dict_offset = fctx.*.dict_content_len - (offset - total_output);
            if (debug) {
                print("X execute_match_copy: dict_copy={}, dict_offset={}\n", .{
                    dict_copy, dict_offset,
                });
                print("{any}\n", .{ fctx.*.dict_content });
            }
            // memcpy(write_ptr, ctx->dict_content + dict_offset, dict_copy);
            // write_ptr += dict_copy;
            lmatch_length -= dict_copy;
            print("TODO: execute_match_copy: breakpoint\n", .{});
            @breakpoint();
        }
    } else if (offset > fctx.*.header.window_size) {
        print("execute_match_copy: Corruption offset={}, windwos_size={}\n",
              .{ offset, fctx.*.header.window_size, });
        return error.Corruption;
    }

    // We must copy byte by byte because the match length might be larger
    // than the offset
    // ex: if the output so far was "abc", a command with offset=3 and
    // match_length=6 would produce "abcabcabc" as the new output

    {
        var j: usize = 0;
        while (j < lmatch_length) : (j += 1) {
            //print("Y execute_match_copy: {} {}, {}, {c}\n", .{
            //    j, lmatch_length, offset, out.*.ptr[out.*.pos - offset]
            //});
            out.*.ptr[out.*.pos] = out.*.ptr[out.*.pos - offset];
            out.*.pos += 1;
        }
    }
}
// //******* END SEQUENCE EXECUTION

fn parse_frame_header(input: *ZStdIStream) !ZStdFrameHeader {
    // "The first header's byte is called the Frame_Header_Descriptor. It tells
    // which other fields are present. Decoding this byte is enough to tell the
    // size of Frame_Header.
    //
    // Bit number   Field name
    // 7-6  Frame_Content_Size_flag
    // 5    Single_Segment_flag
    // 4    Unused_bit
    // 3    Reserved_bit
    // 2    Content_Checksum_flag
    // 1-0  Dictionary_ID_flag"
    const descriptor: u8 = @truncate(u8, try input.read_bits(8));

    // decode frame header descriptor into flags
    const reserved_bit = (descriptor >> 3) & 1;

    if (reserved_bit != 0) {
        return error.InvalidFrameHeaderReservedBit;
    }

    const frame_content_size_flag = descriptor >> 6;
    const dictionary_id_flag = descriptor & 3;

    var frame_header = ZStdFrameHeader {};
    frame_header.single_segment_flag = ((descriptor >> 5) & 1) == 1;
    frame_header.content_checksum_flag = ((descriptor >> 2) & 1) == 1;

    // decode window size
    if (!frame_header.single_segment_flag) {
        // "Provides guarantees on maximum back-reference distance that will be
        // used within compressed data. This information is important for
        // decoders to allocate enough memory.
        //
        // Bit numbers  7-3         2-0
        // Field name   Exponent    Mantissa"
        const window_descriptor: u8 = @truncate(u8, try input.read_bits(8));
        const exponent = window_descriptor >> 3;
        const mantissa = window_descriptor & 7;

        // Use the algorithm from the specification to compute
        // window size
        // https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#window_descriptor
        const window_base: usize = @as(usize, 1) << @truncate(u6, 10 + exponent);
        const window_add: usize = (window_base / 8) * mantissa;
        frame_header.window_size = window_base + window_add;
    }

    // decode dictionary id if it exists
    if (dictionary_id_flag != 0) {
        // "This is a variable size field, which contains the ID of the
        // dictionary required to properly decode the frame. Note that this
        // field is optional. When it's not present, it's up to the caller to
        // make sure it uses the correct dictionary. Format is little-endian."
        const bytes_array: [4]u32 = [_]u32{0, 1, 2, 4};
        const bytes = bytes_array[dictionary_id_flag];

        frame_header.dictionary_id = @truncate(u32, try input.read_bits(bytes * 8));
    } else {
        frame_header.dictionary_id = 0;
    }

    // decode frame content size if it exists
    if (frame_header.single_segment_flag or frame_content_size_flag > 0) {
        // "This is the original (uncompressed) size. This information is
        // optional. The Field_Size is provided according to value of
        // Frame_Content_Size_flag. The Field_Size can be equal to 0 (not
        // present), 1, 2, 4 or 8 bytes. Format is little-endian."
        //
        // if frame_content_size_flag == 0 but single_segment_flag is set, we
        // still have a 1 byte field
        const bytes_array: [4]u32 = [_]u32{1, 2, 4, 8};
        const bytes = bytes_array[frame_content_size_flag];

        frame_header.frame_content_size = try input.read_bits(bytes * 8);
        if (bytes == 2) {
            // "When Field_Size is 2, the offset of 256 is added."
            frame_header.frame_content_size += 256;
        }
    } else {
        frame_header.frame_content_size = 0;
    }

    if (frame_header.single_segment_flag) {
        // "The Window_Descriptor byte is optional. It is absent when
        // Single_Segment_flag is set. In this case, the maximum back-reference
        // distance is the content size itself, which can be any value from 1 to
        // 2^64-1 bytes (16 EB)."
        frame_header.window_size = frame_header.frame_content_size;
    }

    return frame_header;
}

// size_t ZSTD_get_decompressed_size(const void *src, const size_t src_len) {
/// Get the decompressed size of an input stream so memory can be allocated in
/// advance.
/// This implementation assumes `src` points to a single ZSTD-compressed frame
fn ZSTD_get_decompressed_size(src: []const u8) !usize {
    // istream_t in = IO_make_istream(src, src_len);
    var input = ZStdIStream.from_slice(src);

    // Try to get decompressed size from ZSTD frame header
    const magic = @truncate(u32, try input.read_bits(32));
    if (magic != ZSTD_MAGIC) {
        // not a real frame or skippable frame
        return error.InvalidFrameMagic;
    }
    // ZSTD frame
    const fheader = try parse_frame_header(&input);

    if ((fheader.frame_content_size == 0) and (!fheader.single_segment_flag)) {
        // Content size not provided, we can't tell
        return error.NoSizeProvided;
    }
    
    return fheader.frame_content_size;
}


//******* DICTIONARY PARSING
/// The decoded contents of a dictionary so that it doesn't have to be repeated
/// for each frame that uses it
const ZStdDictionary = struct {
    const Self  = @This();

    // Entropy tables
    literals_dtable: HUFTable = HUFTable {},
    ll_dtable: FSETable  = FSETable {},
    ml_dtable: FSETable  = FSETable {},
    of_dtable: FSETable  = FSETable {},

    // Raw content for backreferences
    content: ?[]u8 = null,
    content_size: usize = 0,

    /// Offset history to prepopulate the frame's history
    previous_offsets: [3]u64 = [_]u64{0} ** 3,

    /// Dictionary Id (obviously)
    dictionary_id: u32 = 0,

    fn new_uninit() Self {
        return Self {};
    }
};
// #define DICT_SIZE_ERROR() ERROR("Dictionary size cannot be less than 8 bytes")
// #define NULL_SRC() ERROR("Tried to create dictionary with pointer to null src");

// dictionary_t* create_dictionary() {
//     dictionary_t* dict = calloc(1, sizeof(dictionary_t));
//     if (!dict) {
//         BAD_ALLOC();
//     }
//     return dict;
// }

// void parse_dictionary(dictionary_t *const dict, const void *src,
//                              size_t src_len) {
//     const u8 *byte_src = (const u8 *)src;
//     memset(dict, 0, sizeof(dictionary_t));
//     if (src == NULL) { /* cannot initialize dictionary with null src */
//         NULL_SRC();
//     }
//     if (src_len < 8) {
//         DICT_SIZE_ERROR();
//     }

//     istream_t in = IO_make_istream(byte_src, src_len);

//     const u32 magic_number = IO_read_bits(&in, 32);
//     if (magic_number != 0xEC30A437) {
//         // raw content dict
//         IO_rewind_bits(&in, 32);
//         init_dictionary_content(dict, &in);
//         return;
//     }

//     dict->dictionary_id = IO_read_bits(&in, 32);

//     // "Entropy_Tables : following the same format as the tables in compressed
//     // blocks. They are stored in following order : Huffman tables for literals,
//     // FSE table for offsets, FSE table for match lengths, and FSE table for
//     // literals lengths. It's finally followed by 3 offset values, populating
//     // recent offsets (instead of using {1,4,8}), stored in order, 4-bytes
//     // little-endian each, for a total of 12 bytes. Each recent offset must have
//     // a value < dictionary size."
//     decode_huf_table(&dict->literals_dtable, &in);
//     decode_seq_table(&dict->of_dtable, &in, Offset, FSE);
//     decode_seq_table(&dict->ml_dtable, &in, MatchLength, FSE);
//     decode_seq_table(&dict->ll_dtable, &in, LiteralLength, FSE);

//     // Read in the previous offset history
//     dict->previous_offsets[0] = IO_read_bits(&in, 32);
//     dict->previous_offsets[1] = IO_read_bits(&in, 32);
//     dict->previous_offsets[2] = IO_read_bits(&in, 32);

//     // Ensure the provided offsets aren't too large
//     // "Each recent offset must have a value < dictionary size."
//     for (int i = 0; i < 3; i++) {
//         if (dict->previous_offsets[i] > src_len) {
//             ERROR("Dictionary corrupted");
//         }
//     }

//     // "Content : The rest of the dictionary is its content. The content act as
//     // a "past" in front of data to compress or decompress, so it can be
//     // referenced in sequence commands."
//     init_dictionary_content(dict, &in);
// }

// static void init_dictionary_content(dictionary_t *const dict,
//                                     istream_t *const in) {
//     // Copy in the content
//     dict->content_size = IO_istream_len(in);
//     dict->content = malloc(dict->content_size);
//     if (!dict->content) {
//         BAD_ALLOC();
//     }

//     const u8 *const content = IO_get_read_ptr(in, dict->content_size);

//     memcpy(dict->content, content, dict->content_size);
// }

// Free an allocated dictionary
// void free_dictionary(dictionary_t *const dict) {
//     HUF_free_dtable(&dict->literals_dtable);
//     FSE_free_dtable(&dict->ll_dtable);
//     FSE_free_dtable(&dict->of_dtable);
//     FSE_free_dtable(&dict->ml_dtable);

//     free(dict->content);

//     memset(dict, 0, sizeof(dictionary_t));

//     free(dict);
// }
// //******* END DICTIONARY PARSING
//


test "ZSTD.text.001" {
    const origin = @embedFile("zstd_decompress.zig.zst");
    // try to get the size
    const size = ZSTD_get_decompressed_size(origin) catch 0;
    print("Expected output size: {}\n", .{ size, });
    // allocate space for output
    var output = try gta.alloc(u8, if (size > 0) size else origin.len * 10);
    defer gta.free(output);

    var result = ZSTD_decompress(output, origin, &gta);

    print("result={}, output.len={}, input.len={}\n", .{ result, output.len, origin.len });
    if (result) |r|{
        const goodsz = (size != 0) and (r == size);
        const l = @minimum(1024, r) - 1;
        const h = xxhash.checksum(output[0..r], 0);
        print("Calculated hash: {x}, see *** above, good size {}\n", .{ h, goodsz, });
        // NOTE: We're only dealing with text here -> {s}
        print("Ok: {}, '{s}'\n", .{ r, output[0..l], });
    } else |err| {
        print("Failed: {} {}\n", .{ result, err });
    }
}


// tests generated with "decodecorpus" from zstd/tests...
// test "ZSTD.decodecorpus.001" {
//     const input = @embedFile("z000002.zst");
//     const size = ZSTD_get_decompressed_size(input) catch 0;
//     print("Expected output size: {}\n", .{ size, });
//     var output = try gta.alloc(u8, if (size > 0) size else input.len * 10);
//     // Remember to free
//     defer gta.free(output);

//     var result = ZSTD_decompress(output[0..], input, &gta);

//     print("result={}, output.len={}, input.len={}\n", .{ result, output.len, input.len });
//     if (result) |r|{
//         const goodsz = (size != 0) and (r == size);
//         const l = @minimum(1024, r) - 1;
//         const h = xxhash.checksum(output[0..r], 0);
//         // assert(r == 3387);
//         print("Calculated hash: {x}, good size {}\n", .{ h, goodsz, });
//         // NOTE: We're NOT dealing with text here -> {any}
//         print("Ok: {}, '{any}'\n", .{ r, output[0..l], });
//     } else |err| {
//         print("Failed: {} {}\n", .{ result, err });
//         print("{any}\n", .{ output });
//     }
// }

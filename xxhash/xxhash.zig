const std = @import("std");
const mem = std.mem;
const debug = std.debug;
const print = std.debug.print;


const XXH64_PRIME_1: u64 = 0x9E3779B185EBCA87;
const XXH64_PRIME_2: u64 = 0xC2B2AE3D27D4EB4F;
const XXH64_PRIME_3: u64 = 0x165667B19E3779F9;
const XXH64_PRIME_4: u64 = 0x85EBCA77C2B2AE63;
const XXH64_PRIME_5: u64 = 0x27D4EB2F165667C5;

pub const checksum = xxhash.checksum;

const @"inl" = .{ .modifier = .always_inline };

pub const xxhash = struct {
    const Self = @This();

    seed: u64,
    v1: u64,
    v2: u64,
    v3: u64,
    v4: u64,
    total_len: u64,
    buf: [32]u8,
    buf_used: i64,

    pub fn init(seed: u64) xxhash {
        var hash: Self = undefined;
        hash.seed = seed;
        hash.reset();
        return hash;
    }

    pub fn reset(self: *xxhash) void {
        self.v1 = self.seed +% XXH64_PRIME_1 +% XXH64_PRIME_2;
        self.v2 = self.seed +% XXH64_PRIME_2;
        self.v3 = self.seed;
        self.v4 = self.seed -% XXH64_PRIME_1;
        self.total_len = 0;
        self.buf_used = 0;
    }

    pub fn sum(self: *xxhash) u64 {
        var h64: u64 = 0;
        if (self.total_len >= 32) {
            h64 = rol1(self.v1) +% rol7(self.v2) +% rol12(self.v3) +% rol18(self.v4);
            
            self.v1 *%= XXH64_PRIME_2;
            self.v2 *%= XXH64_PRIME_2;
            self.v3 *%= XXH64_PRIME_2;
            self.v4 *%= XXH64_PRIME_2;

            h64 = (h64^(rol31(self.v1) *% XXH64_PRIME_1)) *% XXH64_PRIME_1 +% XXH64_PRIME_4;
            h64 = (h64^(rol31(self.v2) *% XXH64_PRIME_1)) *% XXH64_PRIME_1 +% XXH64_PRIME_4;
            h64 = (h64^(rol31(self.v3) *% XXH64_PRIME_1)) *% XXH64_PRIME_1 +% XXH64_PRIME_4;
            h64 = (h64^(rol31(self.v4) *% XXH64_PRIME_1)) *% XXH64_PRIME_1 +% XXH64_PRIME_4;

            h64 += self.total_len;
           
        } else {
            h64 = self.seed +% XXH64_PRIME_5 +% self.total_len;
        }
        
        var p: usize = 0;
        var n = self.buf_used;
        
        while (@bitCast(i64, p) <= n - 8): (p += 8) {
            h64 ^= rol31(uint64(self.buf[p..p+8]) *% XXH64_PRIME_2) *% XXH64_PRIME_1;
            h64 = rol27(h64) *% XXH64_PRIME_1 +% XXH64_PRIME_4;
        }
        
        if (@bitCast(i64, p+4) <= n) {
            var sub = self.buf[p..p+4];
            h64 ^= @as(u64, @call(@"inl", uint32, .{sub})) *% XXH64_PRIME_1;
            h64 = rol23(h64) *% XXH64_PRIME_2 +% XXH64_PRIME_3;
            p += 4; 
        }
        
        while (@bitCast(i64, p) < n): (p += 1) {
            h64 ^= @as(u64, self.buf[p]) *% XXH64_PRIME_5;
            h64 = rol11(h64) *% XXH64_PRIME_1;
        }
        
        h64 ^= h64 >> 33;
        h64 *%= XXH64_PRIME_2;
        h64 ^= h64 >> 29;
        h64 *%= XXH64_PRIME_3;
        h64 ^= h64 >> 32;

        return h64;
    }


    pub fn write(self: *xxhash, input: []const u8) usize {
        var n = input.len;
        var m = @bitCast(u64, self.buf_used);

        self.total_len += @as(u64, n);

        var r = self.buf.len - m;

        if (n < r) {
            mem.copy(u8, self.buf[m..], input);
            self.buf_used += @bitCast(i64, input.len);
            return n;
        }

        var p: usize = 0;
        if (m > 0) {
            mem.copy(u8, self.buf[@bitCast(u64, self.buf_used)..], input[0..r]);
            self.buf_used += @bitCast(i64, input.len - r);

            self.v1 = rol31(self.v1 +% uint64(self.buf[0..])  *% XXH64_PRIME_2) *% XXH64_PRIME_1;
            self.v2 = rol31(self.v2 +% uint64(self.buf[8..])  *% XXH64_PRIME_2) *% XXH64_PRIME_1;
            self.v3 = rol31(self.v3 +% uint64(self.buf[16..]) *% XXH64_PRIME_2) *% XXH64_PRIME_1;
            self.v4 = rol31(self.v4 +% uint64(self.buf[24..]) *% XXH64_PRIME_2) *% XXH64_PRIME_1;

            p = r;
            self.buf_used = 0;
        } 

        while (p <= n-32): (p += 32) {
            var sub = input[p..];
            
            self.v1 = rol31(self.v1 +% uint64(sub[0..])  *% XXH64_PRIME_2) *% XXH64_PRIME_1;
            self.v2 = rol31(self.v2 +% uint64(sub[8..])  *% XXH64_PRIME_2) *% XXH64_PRIME_1;
            self.v3 = rol31(self.v3 +% uint64(sub[16..]) *% XXH64_PRIME_2) *% XXH64_PRIME_1;
            self.v4 = rol31(self.v4 +% uint64(sub[24..]) *% XXH64_PRIME_2) *% XXH64_PRIME_1;
        }

        mem.copy(u8, self.buf[@bitCast(usize, self.buf_used)..], input[p..]);
        self.buf_used += @bitCast(i64, input.len - p);
        
        return n;
    }

    pub fn checksum(input: []const u8, seed: u64) u64 {
        var n = input.len;
        var h64: u64 = 0;

        var input2: []const u8 = undefined;
        if (n >= 32) {
            var v1 = seed +% XXH64_PRIME_1 +% XXH64_PRIME_2;
            var v2 = seed +% XXH64_PRIME_2;
            var v3 = seed;
            var v4 = seed -% XXH64_PRIME_1;

            var p: u64 = 0;
            while (p <= n-32): (p += 32) {
                const sub = input[p..];
            
                v1 = rol31(v1 +% @call(@"inl", uint64, .{sub[0..]})
                               *% XXH64_PRIME_2) *% XXH64_PRIME_1;
                v2 = rol31(v2 +% @call(@"inl", uint64, .{sub[8..]})
                               *% XXH64_PRIME_2) *% XXH64_PRIME_1;
                v3 = rol31(v3 +% @call(@"inl", uint64, .{sub[16..]})
                               *% XXH64_PRIME_2) *% XXH64_PRIME_1;
                v4 = rol31(v4 +% @call(@"inl", uint64, .{sub[24..]})
                               *% XXH64_PRIME_2) *% XXH64_PRIME_1;
            }

            h64 = rol1(v1) +% rol7(v2) +% rol12(v3) +% rol18(v4);

            v1 *%= XXH64_PRIME_2;
            v2 *%= XXH64_PRIME_2;
            v3 *%= XXH64_PRIME_2;
            v4 *%= XXH64_PRIME_2;

            h64 = (h64^(rol31(v1) *% XXH64_PRIME_1)) *% XXH64_PRIME_1 +% XXH64_PRIME_4;
            h64 = (h64^(rol31(v2) *% XXH64_PRIME_1)) *% XXH64_PRIME_1 +% XXH64_PRIME_4;
            h64 = (h64^(rol31(v3) *% XXH64_PRIME_1)) *% XXH64_PRIME_1 +% XXH64_PRIME_4;
            h64 = (h64^(rol31(v4) *% XXH64_PRIME_1)) *% XXH64_PRIME_1 +% XXH64_PRIME_4;

            h64 +%= n;

            input2 = input[p..];
            n -= p;
        } else {
            h64 = seed +% XXH64_PRIME_5 +% n; 
            input2 = input[0..];
        }
        
        var p: usize = 0;   
        while (@bitCast(i64,p) <= @bitCast(i64, n) - 8): (p += 8) {
            h64 ^= rol31(@call(@"inl", uint64, .{ input2[p..p+8] })
                             *% XXH64_PRIME_2) *% XXH64_PRIME_1;
            h64 = rol27(h64) *% XXH64_PRIME_1 +% XXH64_PRIME_4;
        }

        if (p+4 <= n) {
            h64 ^= @as(u64, @call(@"inl", uint32, .{ input2[p..p+4] })) *% XXH64_PRIME_1;
            h64 = rol23(h64) *% XXH64_PRIME_2 +% XXH64_PRIME_3;
            p += 4; 
        }

        while (p < n): (p += 1) {
            h64 ^= @as(u64, input2[p]) *% XXH64_PRIME_5;
            h64 = rol11(h64) *% XXH64_PRIME_1;
        }

        h64 ^= h64 >> 33;
        h64 *%= XXH64_PRIME_2;
        h64 ^= h64 >> 29;
        h64 *%= XXH64_PRIME_3;
        h64 ^= h64 >> 32;
        return h64;
    }
};

inline fn uint64(buf: []const u8) u64 {
    return @as(u64, buf[0])
        | @as(u64, buf[1]) <<  8
        | @as(u64, buf[2]) << 16
        | @as(u64, buf[3]) << 24
        | @as(u64, buf[4]) << 32
        | @as(u64, buf[5]) << 40
        | @as(u64, buf[6]) << 48
        | @as(u64, buf[7]) << 56;
}

inline fn uint32(buf: []const u8) u32 {
    return @as(u32, buf[0])
        | @as(u32, buf[1]) <<  8
        | @as(u32, buf[2]) << 16
        | @as(u32, buf[3]) << 24;
}


inline fn rol1(u: u64) u64 {
    return u<<1 | u>>63;
}

inline fn rol7(u: u64) u64 {
    return u<<7 | u>>57;
}

inline fn rol11(u: u64) u64 {
    return u<<11 | u>>53;
}

inline fn rol12(u: u64) u64 {
    return u<<12 | u>>52;
}

inline fn rol18(u: u64) u64 {
    return u<<18 | u>>46;
}

inline fn rol23(u: u64) u64 {
    return u<<23 | u>>41;
}

inline fn rol27(u: u64) u64 {
    return u<<27 | u>>37;
}

inline fn rol31(u: u64) u64 {
    return u<<31 | u>>33;
}

test "xxhash.simple" {
    const XXTest = struct {
        s: []const u8,
        seed: u64 = 0,
        hash: u64 = 0,
    };
    const tests = [_]XXTest {
        XXTest{.s = "xxhash", .seed = 0, .hash = 0x32dd38952c4bc720},
        XXTest{.s = "xxhash", .seed = 20141025, .hash = 0xb559b98d844e0635},
        XXTest{.s = "I want an unsigned 64-bit seed!", .seed = 0, .hash = 0xd4cb0a70a2b8c7c1},
        XXTest{.s = "I want an unsigned 64-bit seed!", .seed = 1, .hash = 0xce5087f12470d961},
    };
    for (tests) |*t, i| {
        const h = xxhash.checksum(t.*.s, t.*.seed);
        var hasher = xxhash.init(t.*.seed);
        _ = hasher.write(t.*.s);
        const hh = hasher.sum();
        print("\n[{d:2}] h={x},hh={x} vs {x} '{s}' w/seed {d}",
              .{ i, h, hh, t.*.hash, t.*.s, t.*.seed });
        debug.assert(h == hh);
        debug.assert(h == t.*.hash);
    }
    print("\n", .{});
}

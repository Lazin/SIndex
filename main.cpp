#include <iostream>
#include <algorithm>
#include <memory>
#include <atomic>
#include <mutex>
#include <deque>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <ctime>
#include <vector>
#include <random>
#include <iterator>
#include <sstream>

#define BOOST_THROW_EXCEPTION(x) throw x;

typedef std::uint32_t u32;
typedef std::int32_t  i32;
typedef std::uint64_t u64;
typedef std::int64_t  i64;

enum aku_Status {
    AKU_SUCCESS,
    AKU_EBAD_ARG,
    AKU_EBAD_DATA,
};

enum {
    AKU_LIMITS_MAX_SNAME = 2048,
    AKU_LIMITS_MAX_TAGS  = 256,
};

/** Namespace class to store all parsing related things.
  */
struct SeriesParser {
    /** Convert input string to normal form.
      * In normal form metric name is followed by the list of key
      * value pairs in alphabetical order. All keys should be unique and
      * separated from metric name and from each other by exactly one space.
      * @param begin points to the begining of the input string
      * @param end points to the to the end of the string
      * @param out_begin points to the begining of the output buffer (should be not less then input buffer)
      * @param out_end points to the end of the output buffer
      * @param keystr_begin points to the begining of the key string (string with key-value pairs)
      * @return AKU_SUCCESS if everything is OK, error code otherwise
      */
    static aku_Status to_normal_form(const char* begin, const char* end, char* out_begin,
                                     char* out_end, const char** keystr_begin,
                                     const char** keystr_end);
};

//                         //
//      Series Parser      //
//                         //

//! Move pointer to the of the whitespace, return this pointer or end on error
static const char* skip_space(const char* p, const char* end) {
    while(p < end && (*p == ' ' || *p == '\t')) {
        p++;
    }
    return p;
}

static const char* copy_until(const char* begin, const char* end, const char pattern, char** out) {
    char* it_out = *out;
    while(begin < end) {
        *it_out = *begin;
        it_out++;
        begin++;
        if (*begin == pattern) {
            break;
        }
    }
    *out = it_out;
    return begin;
}

//! Move pointer to the beginning of the next tag, return this pointer or end on error
static const char* skip_tag(const char* p, const char* end, bool *error) {
    // skip until '='
    while(p < end && *p != '=' && *p != ' ' && *p != '\t') {
        p++;
    }
    if (p == end || *p != '=') {
        *error = true;
        return end;
    }
    // skip until ' '
    const char* c = p;
    while(c < end && *c != ' ') {
        c++;
    }
    *error = c == p;
    return c;
}

aku_Status SeriesParser::to_normal_form(const char* begin, const char* end,
                                        char* out_begin, char* out_end,
                                        const char** keystr_begin,
                                        const char** keystr_end)
{
    // Verify args
    if (end < begin) {
        return AKU_EBAD_ARG;
    }
    if (out_end < out_begin) {
        return AKU_EBAD_ARG;
    }
    int series_name_len = end - begin;
    if (series_name_len > AKU_LIMITS_MAX_SNAME) {
        return AKU_EBAD_DATA;
    }
    if (series_name_len > (out_end - out_begin)) {
        return AKU_EBAD_ARG;
    }

    char* it_out = out_begin;
    const char* it = begin;
    // Get metric name
    it = skip_space(it, end);
    it = copy_until(it, end, ' ', &it_out);
    it = skip_space(it, end);

    if (it == end) {
        // At least one tag should be specified
        return AKU_EBAD_DATA;
    }

    *keystr_begin = it_out;

    // Get pointers to the keys
    const char* tags[AKU_LIMITS_MAX_TAGS];
    auto ix_tag = 0u;
    bool error = false;
    while(it < end && ix_tag < AKU_LIMITS_MAX_TAGS) {
        tags[ix_tag] = it;
        it = skip_tag(it, end, &error);
        it = skip_space(it, end);
        if (!error) {
            ix_tag++;
        } else {
            break;
        }
    }
    if (error) {
        // Bad string
        return AKU_EBAD_DATA;
    }
    if (ix_tag == 0) {
        // User should specify at least one tag
        return AKU_EBAD_DATA;
    }

    auto sort_pred = [tags, end](const char* lhs, const char* rhs) {
        // lhs should be always less thenn rhs
        auto lenl = 0u;
        auto lenr = 0u;
        if (lhs < rhs) {
            lenl = rhs - lhs;
            lenr = end - rhs;
        } else {
            lenl = end - lhs;
            lenr = lhs - rhs;
        }
        auto it = 0u;
        while(true) {
            if (it >= lenl || it >= lenr) {
                return it < lenl;
            }
            if (lhs[it] == '=' || rhs[it] == '=') {
                return lhs[it] == '=';
            }
            if (lhs[it] < rhs[it]) {
                return true;
            } else if (lhs[it] > rhs[it]) {
                return false;
            }
            it++;
        }
    };

    std::sort(tags, tags + ix_tag, std::ref(sort_pred));  // std::sort can't move from predicate if predicate is a rvalue
                                                          // nor it can pass predicate by reference
    // Copy tags to output string
    for (auto i = 0u; i < ix_tag; i++) {
        // insert space
        *it_out++ = ' ';
        // insert tag
        const char* tag = tags[i];
        copy_until(tag, end, ' ', &it_out);
    }
    *keystr_begin = skip_space(*keystr_begin, out_end);
    *keystr_end = it_out;
    return AKU_SUCCESS;
}

//                       //
//      String Pool      //
//                       //

typedef std::pair<const char*, u32> StringT;

StringT tostrt(const char* p) {
    return std::make_pair(p, strlen(p));
}

StringT tostrt(std::string const& s) {
    return std::make_pair(s.data(), s.size());
}

std::string fromstrt(StringT s) {
    return std::string(s.first, s.first + s.second);
}

class StringPool {
public:
    const u64 MAX_BIN_SIZE = AKU_LIMITS_MAX_SNAME * 0x1000;  // 8Mb

    std::deque<std::vector<char>> pool;
    mutable std::mutex            pool_mutex;
    std::atomic<size_t>           counter;

    StringPool();
    StringPool(StringPool const&) = delete;
    StringPool& operator=(StringPool const&) = delete;

    /**
     * @brief add value to string pool
     * @param begin is a pointer to the begining of the string
     * @param end is a pointer to the next character after the end of the string
     * @return Z-order encoded address of the string (0 in case of error)
     */
    u64 add(const char* begin, const char* end);

    /**
     * @brief str returns string representation
     * @param bits is a Z-order encoded position in the string buffer
     * @return 0-copy string representation (or empty string)
     */
    StringT str(u64 bits) const;

    //! Get number of stored strings atomically
    size_t size() const;

    size_t mem_used() const;
};

u64 splitBits(u32 val) {
    u64 x = val & 0x1fffff;
    x = (x | x << 32) & 0x001f00000000ffff;
    x = (x | x << 16) & 0x001f0000ff0000ff;
    x = (x | x <<  8) & 0x100f00f00f00f00f;
    x = (x | x <<  4) & 0x10c30c30c30c30c3;
    x = (x | x <<  2) & 0x1249249249249249;
    return x;
}

u32 compactBits(u64 x) {
    x &= 0x1249249249249249;
    x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
    x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
    x = (x ^ (x >> 8)) & 0x001f0000ff0000ff;
    x = (x ^ (x >>16)) & 0x001f00000000ffff;
    x = (x ^ (x >>32)) & 0x00000000001fffff;
    return static_cast<u32>(x);
}

u64 encodeZorder(u32 x, u32 y) {
    return splitBits(x) | (splitBits(y) << 1);
}

u32 decodeZorderX(u64 bits) {
    return compactBits(bits);
}

u32 decodeZorderY(u64 bits) {
    return compactBits(bits >> 1);
}

StringPool::StringPool()
    : counter{0}
{
}

u64 StringPool::add(const char* begin, const char* end) {
    assert(begin < end);
    std::lock_guard<std::mutex> guard(pool_mutex);
    if (pool.empty()) {
        pool.emplace_back();
        pool.back().reserve(MAX_BIN_SIZE);
    }
    auto size = static_cast<u64>(end - begin);
    if (size == 0) {
        return 0;
    }
    size += 1;  // 1 is for 0 character
    u32 bin_index = static_cast<u32>(pool.size()); // bin index is 1-based
    std::vector<char>* bin = &pool.back();
    if (bin->size() + size > MAX_BIN_SIZE) {
        // New bin
        pool.emplace_back();
        bin = &pool.back();
        bin->reserve(MAX_BIN_SIZE);
        bin_index = static_cast<u32>(pool.size());
    }
    u32 offset = static_cast<u32>(bin->size()); // offset is 0-based
    for(auto i = begin; i < end; i++) {
        bin->push_back(*i);
    }
    bin->push_back('\0');
    std::atomic_fetch_add(&counter, 1ul);
    return bin_index*MAX_BIN_SIZE + offset;
}

StringT StringPool::str(u64 bits) const {
    u64 ix     = bits / MAX_BIN_SIZE;
    u64 offset = bits % MAX_BIN_SIZE;
    assert(ix != 0);
    std::lock_guard<std::mutex> guard(pool_mutex);
    if (ix <= pool.size()) {
        std::vector<char> const* bin = &pool.at(ix - 1);
        if (offset < bin->size()) {
            const char* pstr = bin->data() + offset;
            return std::make_pair(pstr, std::strlen(pstr));
        }
    }
    return std::make_pair(nullptr, 0);
}

size_t StringPool::size() const {
    return std::atomic_load(&counter);
}

size_t StringPool::mem_used() const {
    size_t res = 0;
    std::lock_guard<std::mutex> guard(pool_mutex);
    for (auto const& bin: pool) {
        res += bin.size();
    }
    return res;
}


//               //
//  StringTools  //
//               //

struct StringTools {
    static size_t hash(StringT str);
    static bool equal(StringT lhs, StringT rhs);

    typedef std::unordered_map<StringT, u64, decltype(&StringTools::hash),
                               decltype(&StringTools::equal)>
        TableT;

    typedef std::unordered_set<StringT, decltype(&StringTools::hash), decltype(&StringTools::equal)>
        SetT;

    typedef std::unordered_map<StringT, SetT, decltype(&StringTools::hash), decltype(&StringTools::equal)> L2TableT;

    typedef  std::unordered_map<StringT, L2TableT, decltype(&StringTools::hash), decltype(&StringTools::equal)> L3TableT;

    //! Inverted table type (id to string mapping)
    typedef std::unordered_map<u64, StringT> InvT;

    static TableT create_table(size_t size);

    static SetT create_set(size_t size);

    static L2TableT create_l2_table(size_t size_hint);

    static L3TableT create_l3_table(size_t size_hint);
};

StringTools::L2TableT StringTools::create_l2_table(size_t size_hint) {
    return L2TableT(size_hint, &StringTools::hash, &StringTools::equal);
}

StringTools::L3TableT StringTools::create_l3_table(size_t size_hint) {
    return L3TableT(size_hint, &StringTools::hash, &StringTools::equal);
}

size_t StringTools::hash(StringT str) {
    // implementation of Dan Bernstein's djb2
    const char* begin = str.first;
    int len = str.second;
    const char* end = begin + len;
    size_t hash = 5381;
    size_t c;
    while (begin < end) {
        c = static_cast<size_t>(*begin++);
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }
    return hash;
}

bool StringTools::equal(StringT lhs, StringT rhs) {
    if (lhs.second != rhs.second) {
        return false;
    }
    return std::equal(lhs.first, lhs.first + lhs.second, rhs.first);
}

StringTools::TableT StringTools::create_table(size_t size) {
    return TableT(size, &StringTools::hash, &StringTools::equal);
}

StringTools::SetT StringTools::create_set(size_t size) {
    return SetT(size, &StringTools::hash, &StringTools::equal);
}

//                  //
//  Hash-fn family  //
//                  //

//! Family of 4-universal hash functions
struct HashFnFamily {
    const u32 N;
    const u32 K;
    //! Tabulation based hash fn used, N tables should be generated using RNG in c-tor
    std::vector<std::vector<unsigned short>> table_;

    //! C-tor. N - number of different hash functions, K - number of values (should be a power of two)
    HashFnFamily(u32 N, u32 K);

    //! Calculate hash value in range [0, K)
    u32 hash(int ix, u64 key) const;

private:
    u32 hash32(int ix, u32 key) const;
};

static u32 combine(u32 hi, u32 lo) {
    return (u32)(2 - (int)hi + (int)lo);
}

HashFnFamily::HashFnFamily(u32 N, u32 K)
    : N(N)
    , K(K)
{
    // N should be odd
    if (N % 2 == 0) {
        std::runtime_error err("invalid argument N (should be odd)");
        BOOST_THROW_EXCEPTION(err);
    }
    // K should be a power of two
    auto mask = K-1;
    if ((mask&K) != 0) {
        std::runtime_error err("invalid argument K (should be a power of two)");
        BOOST_THROW_EXCEPTION(err);
    }
    // Generate tables
    std::random_device randdev;
    std::mt19937 generator(randdev());
    std::uniform_int_distribution<> distribution;
    for (u32 i = 0; i < N; i++) {
        std::vector<unsigned short> col;
        auto mask = K-1;
        for (int j = 0; j < 0x10000; j++) {
            int value = distribution(generator);
            col.push_back((u32)mask&value);
        }
        table_.push_back(col);
    }
}

u32 HashFnFamily::hash(int ix, u64 key) const {
    auto hi32 = key >> 32;
    auto lo32 = key & 0xFFFFFFFF;
    auto hilo = combine(hi32, lo32);

    auto hi32hash = hash32(ix, hi32);
    auto lo32hash = hash32(ix, lo32);
    auto hilohash = hash32(ix, hilo);

    return hi32hash ^ lo32hash ^ hilohash;
}

u32 HashFnFamily::hash32(int ix, u32 key) const {
    auto hi16 = key >> 16;
    auto lo16 = key & 0xFFFF;
    auto hilo = combine(hi16, lo16) & 0xFFFF;
    return table_[ix][lo16] ^ table_[ix][hi16] ^ table_[ix][hilo];
}

//             //
//  CM-sketch  //
//             //

namespace {
//! Base 128 encoded integer
template <class TVal> class Base128Int {
    TVal                  value_;
    typedef unsigned char byte_t;
    typedef byte_t*       byte_ptr;

public:
    Base128Int(TVal val)
        : value_(val) {}

    Base128Int()
        : value_() {}

    /** Read base 128 encoded integer from the binary stream
      * FwdIter - forward iterator.
      */
    const unsigned char* get(const unsigned char* begin, const unsigned char* end) {
        assert(begin < end);

        auto                 acc = TVal();
        auto                 cnt = TVal();
        const unsigned char* p   = begin;

        while (true) {
            if (p == end) {
                return begin;
            }
            auto i = static_cast<byte_t>(*p & 0x7F);
            acc |= TVal(i) << cnt;
            if ((*p++ & 0x80) == 0) {
                break;
            }
            cnt += 7;
        }
        value_ = acc;
        return p;
    }

    /** Write base 128 encoded integer to the binary stream.
      * @returns 'begin' on error, iterator to next free region otherwise
      */
    void put(std::vector<char>& vec) const {
        TVal           value = value_;
        unsigned char p;
        while (true) {
            p = value & 0x7F;
            value >>= 7;
            if (value != 0) {
                p |= 0x80;
                vec.push_back(static_cast<char>(p));
            } else {
                vec.push_back(static_cast<char>(p));
                break;
            }
        }
    }

    //! turn into integer
    operator TVal() const { return value_; }
};

//! Base128 encoder
struct Base128StreamWriter {
    // underlying memory region
    std::vector<char>* buffer_;

    Base128StreamWriter(std::vector<char>& buffer)
        : buffer_(&buffer)
    {}

    Base128StreamWriter(Base128StreamWriter const& other)
        : buffer_(other.buffer_)
    {}

    Base128StreamWriter& operator = (Base128StreamWriter const& other) {
        buffer_ = other.buffer_;
    }

    void reset(std::vector<char>& buffer) {
        buffer_ = &buffer;
    }

    bool empty() const { return buffer_->empty(); }

    //! Put value into stream.
    template <class TVal> bool put(TVal value) {
        Base128Int<TVal> val(value);
        val.put(*buffer_);
        return true;
    }
};

//! Base128 decoder
struct Base128StreamReader {
    const unsigned char* pos_;
    const unsigned char* end_;

    Base128StreamReader(const unsigned char* begin, const unsigned char* end)
        : pos_(begin)
        , end_(end) {}

    Base128StreamReader(Base128StreamReader const& other)
        : pos_(other.pos_)
        , end_(other.end_)
    {
    }

    Base128StreamReader& operator = (Base128StreamReader const& other) {
        if (&other == this) {
            return *this;
        }
        pos_ = other.pos_;
        end_ = other.end_;
    }

    template <class TVal> TVal next() {
        Base128Int<TVal> value;
        auto             p = value.get(pos_, end_);
        if (p == pos_) {
            std::cerr << "Base128Stream read error" << std::endl;
            std::terminate();
        }
        pos_ = p;
        return static_cast<TVal>(value);
    }
};

template <class Stream, typename TVal> struct DeltaStreamWriter {
    Stream* stream_;
    TVal   prev_;

    template<class Substream>
    DeltaStreamWriter(Substream& stream)
        : stream_(&stream)
        , prev_{}
    {}

    DeltaStreamWriter(DeltaStreamWriter const& other)
        : stream_(other.stream_)
        , prev_(other.prev_)
    {
    }

    DeltaStreamWriter& operator = (DeltaStreamWriter const& other) {
        if (this == &other) {
            return *this;
        }
        stream_ = other.stream_;
        prev_   = other.prev_;
        return *this;
    }

    bool put(TVal value) {
        auto result = stream_->put(static_cast<TVal>(value) - prev_);
        prev_       = value;
        return result;
    }
};


template <class Stream, typename TVal> struct DeltaStreamReader {
    Stream* stream_;
    TVal   prev_;

    DeltaStreamReader(Stream& stream)
        : stream_(&stream)
        , prev_() {}

    DeltaStreamReader(DeltaStreamReader const& other)
        : stream_(other.stream_)
        , prev_(other.prev_)
    {
    }

    DeltaStreamReader& operator = (DeltaStreamReader const& other) {
        if (&other == this) {
            return *this;
        }
        stream_ = other.stream_;
        prev_   = other.prev_;
        return *this;
    }

    TVal next() {
        TVal delta = stream_->template next<TVal>();
        TVal value = prev_ + delta;
        prev_      = value;
        return value;
    }
};

}

// Iterator for compressed PList

class CompressedPListConstIterator {
    size_t card_;
    Base128StreamReader reader_;
    DeltaStreamReader<Base128StreamReader, u64> delta_;
    size_t pos_;
    u64 curr_;
public:
    typedef u64 value_type;

    CompressedPListConstIterator(std::vector<char> const& vec, size_t c)
        : card_(c)
        , reader_(reinterpret_cast<const unsigned char*>(vec.data()),
                  reinterpret_cast<const unsigned char*>(vec.data() + vec.size()))
        , delta_(reader_)
        , pos_(0)
    {
        if (pos_ < card_) {
            curr_ = delta_.next();
        }
    }

    /**
     * @brief Create iterator pointing to the end of the sequence
     */
    CompressedPListConstIterator(std::vector<char> const& vec, size_t c, bool)
        : card_(c)
        , reader_(reinterpret_cast<const unsigned char*>(vec.data()),
                  reinterpret_cast<const unsigned char*>(vec.data() + vec.size()))
        , delta_(reader_)
        , pos_(c)
        , curr_()
    {
    }

    CompressedPListConstIterator(CompressedPListConstIterator const& other)
        : card_(other.card_)
        , reader_(other.reader_)
        , delta_(other.delta_)
        , pos_(other.pos_)
        , curr_(other.curr_)
    {
    }

    CompressedPListConstIterator& operator = (CompressedPListConstIterator const& other) {
        if (this == &other) {
            return *this;
        }
        card_ = other.card_;
        reader_ = other.reader_;
        delta_ = other.delta_;
        pos_ = other.pos_;
        curr_ = other.curr_;
        return *this;
    }

    u64 operator * () const {
        return curr_;
    }

    CompressedPListConstIterator& operator ++ () {
        pos_++;
        if (pos_ < card_) {
            curr_ = delta_.next();
        }
        return *this;
    }

    bool operator == (CompressedPListConstIterator const& other) const {
        return pos_ == other.pos_;
    }

    bool operator != (CompressedPListConstIterator const& other) const {
        return pos_ != other.pos_;
    }
};

namespace std {
    template<>
    struct iterator_traits<CompressedPListConstIterator> {
        typedef u64 value_type;
        typedef forward_iterator_tag iterator_category;
    };
}

class CompressedPList {
    std::vector<char> buffer_;
    Base128StreamWriter writer_;
    DeltaStreamWriter<Base128StreamWriter, u64> delta_;
    size_t cardinality_;
    bool moved_;
public:

    typedef u64 value_type;

    CompressedPList()
        : writer_(buffer_)
        , delta_(writer_)
        , cardinality_(0)
        , moved_(false)
    {
    }

    CompressedPList(CompressedPList const& other)
        : buffer_(other.buffer_)
        , writer_(buffer_)
        , delta_(writer_)
        , cardinality_(other.cardinality_)
        , moved_(false)
    {
        assert(!other.moved_);
    }

    CompressedPList& operator = (CompressedPList && other) {
        assert(!other.moved_);
        if (this == &other) {
            return *this;
        }
        other.moved_ = true;
        buffer_.swap(other.buffer_);
        // we don't need to assign writer_ since it contains pointer to buffer_
        // already
        // delta already contain correct pointer to writer_ we only need to
        // update prev_ field
        delta_.prev_ = other.delta_.prev_;
        cardinality_ = other.cardinality_;
        return *this;
    }

    CompressedPList(CompressedPList && other)
        : buffer_(std::move(other.buffer_))
        , writer_(buffer_)
        , delta_(writer_)
        , cardinality_(other.cardinality_)
        , moved_(false)
    {
        assert(!other.moved_);
        other.moved_ = true;
    }

    CompressedPList& operator = (CompressedPList const& other) = delete;

    void add(u64 x) {
        assert(!moved_);
        delta_.put(x);
        cardinality_++;
    }

    void push_back(u64 x) {
        assert(!moved_);
        add(x);
    }

    size_t getSizeInBytes() const {
        assert(!moved_);
        return buffer_.capacity();
    }

    size_t cardinality() const {
        assert(!moved_);
        return cardinality_;
    }

    CompressedPList operator & (CompressedPList const& other) const {
        assert(!moved_);
        CompressedPList result;
        std::set_intersection(begin(), end(), other.begin(), other.end(),
                              std::back_inserter(result));
        return result;
    }

    CompressedPList operator | (CompressedPList const& other) const {
        assert(!moved_);
        CompressedPList result;
        std::set_union(begin(), end(), other.begin(), other.end(),
                       std::back_inserter(result));
        return result;
    }

    CompressedPList operator ^ (CompressedPList const& other) const {
        assert(!moved_);
        CompressedPList result;
        std::set_difference(begin(), end(), other.begin(), other.end(),
                            std::back_inserter(result));
        return result;
    }


    CompressedPListConstIterator begin() const {
        assert(!moved_);
        return CompressedPListConstIterator(buffer_, cardinality_);
    }

    CompressedPListConstIterator end() const {
        assert(!moved_);
        return CompressedPListConstIterator(buffer_, cardinality_, false);
    }
};

class PList {
    std::vector<u64> lst_;
public:
    void add(u64 x) { lst_.push_back(x); }
    size_t getSizeInBytes() const {
        return lst_.capacity() * sizeof(u64);
    }

    PList operator & (PList const& other) const {
        PList result;
        std::set_intersection(lst_.begin(), lst_.end(), other.lst_.begin(), other.lst_.end(),
                              std::back_inserter(result.lst_));
        return result;
    }
    size_t cardinality() const {
        return lst_.size();
    }
    std::vector<u64>::const_iterator begin() const {
        return lst_.begin();
    }
    std::vector<u64>::const_iterator end() const {
        return lst_.end();
    }
};

class CMSketch {
    //typedef Roaring TVal;
    //typedef PList TVal;
    typedef CompressedPList TVal;
    std::vector<std::vector<TVal>> table_;
    HashFnFamily hashfn_;
    const u32 N;
    const u32 M;
public:
    CMSketch(u32 N, u32 M) : hashfn_(N, M), N(N), M(M) {
        table_.resize(N);
        for (auto& row: table_) {
            row.resize(M);
        }
    }

    void add(u64 value) {
        for (u32 i = 0; i < N; i++) {
            // calculate hash from id to K
            u32 hash = hashfn_.hash(i, value);
            table_[i][hash].add((u32)value);
        }
    }

    void add(u64 key, u64 value) {
        for (u32 i = 0; i < N; i++) {
            // calculate hash from id to K
            u32 hash = hashfn_.hash(i, key);
            table_[i][hash].add(value);
        }
    }

    size_t get_size_in_bytes() const {
        size_t sum = 0;
        for (auto const& row: table_) {
            for (auto const& bmp: row) {
                sum += bmp.getSizeInBytes();
            }
        }
        return sum;
    }

    TVal extract(u64 value) const {
        std::vector<const TVal*> inputs;
        for (u32 i = 0; i < N; i++) {
            // calculate hash from id to K
            u32 hash = hashfn_.hash(i, value);
            inputs.push_back(&table_[i][hash]);
        }
        return *inputs[0] & *inputs[1] & *inputs[2];
    }
};


std::vector<std::string> sample_lines = {
    "cpu.user OS=Ubuntu_14.04 arch=x64 host=192.168.0.0 instance-type=m3.large rack=86 region=eu-central-1 team=NJ",
    "cpu.sys OS=Ubuntu_14.04 arch=x64 host=192.168.0.0 instance-type=m3.large rack=86 region=eu-central-1 team=NJ",
    "cpu.real OS=Ubuntu_14.04 arch=x64 host=192.168.0.0 instance-type=m3.large rack=86 region=eu-central-1 team=NJ",
    "idle OS=Ubuntu_14.04 arch=x64 host=192.168.0.0 instance-type=m3.large rack=86 region=eu-central-1 team=NJ",
    "mem.commit OS=Ubuntu_14.04 arch=x64 host=192.168.0.0 instance-type=m3.large rack=86 region=eu-central-1 team=NJ",
    "mem.virt OS=Ubuntu_14.04 arch=x64 host=192.168.0.0 instance-type=m3.large rack=86 region=eu-central-1 team=NJ",
    "iops OS=Ubuntu_14.04 arch=x64 host=192.168.0.0 instance-type=m3.large rack=86 region=eu-central-1 team=NJ",
    "tcp.packets.in OS=Ubuntu_14.04 arch=x64 host=192.168.0.0 instance-type=m3.large rack=86 region=eu-central-1 team=NJ",
    "tcp.packets.out OS=Ubuntu_14.04 arch=x64 host=192.168.0.0 instance-type=m3.large rack=86 region=eu-central-1 team=NJ",
    "tcp.ret OS=Ubuntu_14.04 arch=x64 host=192.168.0.0 instance-type=m3.large rack=86 region=eu-central-1 team=NJ",
    "cpu.user OS=Ubuntu_16.04 arch=x64 host=192.168.0.1 instance-type=m4.2xlarge rack=96 region=eu-central-1 team=NJ",
    "cpu.sys OS=Ubuntu_16.04 arch=x64 host=192.168.0.1 instance-type=m4.2xlarge rack=96 region=eu-central-1 team=NJ",
    "cpu.real OS=Ubuntu_16.04 arch=x64 host=192.168.0.1 instance-type=m4.2xlarge rack=96 region=eu-central-1 team=NJ",
    "idle OS=Ubuntu_16.04 arch=x64 host=192.168.0.1 instance-type=m4.2xlarge rack=96 region=eu-central-1 team=NJ",
    "mem.commit OS=Ubuntu_16.04 arch=x64 host=192.168.0.1 instance-type=m4.2xlarge rack=96 region=eu-central-1 team=NJ",
    "mem.virt OS=Ubuntu_16.04 arch=x64 host=192.168.0.1 instance-type=m4.2xlarge rack=96 region=eu-central-1 team=NJ",
    "iops OS=Ubuntu_16.04 arch=x64 host=192.168.0.1 instance-type=m4.2xlarge rack=96 region=eu-central-1 team=NJ",
    "tcp.packets.in OS=Ubuntu_16.04 arch=x64 host=192.168.0.1 instance-type=m4.2xlarge rack=96 region=eu-central-1 team=NJ",
    "tcp.packets.out OS=Ubuntu_16.04 arch=x64 host=192.168.0.1 instance-type=m4.2xlarge rack=96 region=eu-central-1 team=NJ",
    "tcp.ret OS=Ubuntu_16.04 arch=x64 host=192.168.0.1 instance-type=m4.2xlarge rack=96 region=eu-central-1 team=NJ",
    "cpu.user OS=Ubuntu_14.04 arch=x64 host=192.168.0.2 instance-type=m4.large rack=90 region=eu-central-1 team=NJ",
    "cpu.sys OS=Ubuntu_14.04 arch=x64 host=192.168.0.2 instance-type=m4.large rack=90 region=eu-central-1 team=NJ",
    "cpu.real OS=Ubuntu_14.04 arch=x64 host=192.168.0.2 instance-type=m4.large rack=90 region=eu-central-1 team=NJ",
    "idle OS=Ubuntu_14.04 arch=x64 host=192.168.0.2 instance-type=m4.large rack=90 region=eu-central-1 team=NJ",
    "mem.commit OS=Ubuntu_14.04 arch=x64 host=192.168.0.2 instance-type=m4.large rack=90 region=eu-central-1 team=NJ",
    "mem.virt OS=Ubuntu_14.04 arch=x64 host=192.168.0.2 instance-type=m4.large rack=90 region=eu-central-1 team=NJ",
    "iops OS=Ubuntu_14.04 arch=x64 host=192.168.0.2 instance-type=m4.large rack=90 region=eu-central-1 team=NJ",
    "tcp.packets.in OS=Ubuntu_14.04 arch=x64 host=192.168.0.2 instance-type=m4.large rack=90 region=eu-central-1 team=NJ",
    "tcp.packets.out OS=Ubuntu_14.04 arch=x64 host=192.168.0.2 instance-type=m4.large rack=90 region=eu-central-1 team=NJ",
    "tcp.ret OS=Ubuntu_14.04 arch=x64 host=192.168.0.2 instance-type=m4.large rack=90 region=eu-central-1 team=NJ",
    "cpu.user OS=Ubuntu_16.04 arch=x64 host=192.168.0.3 instance-type=m4.2xlarge rack=77 region=us-east-1 team=NJ",
    "cpu.sys OS=Ubuntu_16.04 arch=x64 host=192.168.0.3 instance-type=m4.2xlarge rack=77 region=us-east-1 team=NJ",
    "cpu.real OS=Ubuntu_16.04 arch=x64 host=192.168.0.3 instance-type=m4.2xlarge rack=77 region=us-east-1 team=NJ",
    "idle OS=Ubuntu_16.04 arch=x64 host=192.168.0.3 instance-type=m4.2xlarge rack=77 region=us-east-1 team=NJ",
    "mem.commit OS=Ubuntu_16.04 arch=x64 host=192.168.0.3 instance-type=m4.2xlarge rack=77 region=us-east-1 team=NJ",
    "mem.virt OS=Ubuntu_16.04 arch=x64 host=192.168.0.3 instance-type=m4.2xlarge rack=77 region=us-east-1 team=NJ",
    "iops OS=Ubuntu_16.04 arch=x64 host=192.168.0.3 instance-type=m4.2xlarge rack=77 region=us-east-1 team=NJ",
    "tcp.packets.in OS=Ubuntu_16.04 arch=x64 host=192.168.0.3 instance-type=m4.2xlarge rack=77 region=us-east-1 team=NJ",
    "tcp.packets.out OS=Ubuntu_16.04 arch=x64 host=192.168.0.3 instance-type=m4.2xlarge rack=77 region=us-east-1 team=NJ",
    "tcp.ret OS=Ubuntu_16.04 arch=x64 host=192.168.0.3 instance-type=m4.2xlarge rack=77 region=us-east-1 team=NJ",
    "cpu.user OS=Ubuntu_14.04 arch=x64 host=192.168.0.4 instance-type=m3.large rack=63 region=us-east-1 team=NY",
    "cpu.sys OS=Ubuntu_14.04 arch=x64 host=192.168.0.4 instance-type=m3.large rack=63 region=us-east-1 team=NY",
    "cpu.real OS=Ubuntu_14.04 arch=x64 host=192.168.0.4 instance-type=m3.large rack=63 region=us-east-1 team=NY",
    "idle OS=Ubuntu_14.04 arch=x64 host=192.168.0.4 instance-type=m3.large rack=63 region=us-east-1 team=NY",
    "mem.commit OS=Ubuntu_14.04 arch=x64 host=192.168.0.4 instance-type=m3.large rack=63 region=us-east-1 team=NY",
    "mem.virt OS=Ubuntu_14.04 arch=x64 host=192.168.0.4 instance-type=m3.large rack=63 region=us-east-1 team=NY",
    "iops OS=Ubuntu_14.04 arch=x64 host=192.168.0.4 instance-type=m3.large rack=63 region=us-east-1 team=NY",
};

class PerfTimer {
public:
    PerfTimer();
    void   restart();
    double elapsed() const;

private:
    timespec _start_time;
};

PerfTimer::PerfTimer() {
    clock_gettime(CLOCK_MONOTONIC_RAW, &_start_time);
}

void PerfTimer::restart() {
    clock_gettime(CLOCK_MONOTONIC_RAW, &_start_time);
}

double PerfTimer::elapsed() const {
    timespec curr;
    clock_gettime(CLOCK_MONOTONIC_RAW, &curr);
    return double(curr.tv_sec - _start_time.tv_sec) +
           double(curr.tv_nsec - _start_time.tv_nsec)/1000000000.0;
}

static void write_tags(const char* begin, const char* end, CMSketch* dest_sketch, u64 id) {
    const char* tag_begin = begin;
    const char* tag_end = begin;
    bool err = false;
    while(!err && tag_begin != end) {
        tag_begin = skip_space(tag_begin, end);
        tag_end = tag_begin;
        tag_end = skip_tag(tag_end, end, &err);
        auto tagpair = std::make_pair(tag_begin, static_cast<u32>(tag_end - tag_begin));
        u64 hash = StringTools::hash(tagpair);
        dest_sketch->add(hash, id);
        tag_begin = tag_end;
    }
    if (err) {
        std::cerr << "Failure" << std::endl;
        std::abort();
    }
}

static StringT skip_metric_name(const char* begin, const char* end) {
    const char* p = begin;
    // skip metric name
    p = skip_space(p, end);
    if (p == end) {
        return std::make_pair(nullptr, 0);
    }
    const char* m = p;
    while(*p != ' ') {
        p++;
    }
    return std::make_pair(m, p - m);
}


class MetricName {
    std::string name_;
public:
    MetricName(const char* begin, const char* end)
        : name_(begin, end)
    {
    }

    MetricName(const char* str)
        : name_(str)
    {
    }

    StringT get_value() const {
        return std::make_pair(name_.data(), name_.size());
    }

    bool check(const char* begin, const char* end) const {
        auto name = skip_metric_name(begin, end);
        if (name.second == 0) {
            return false;
        }
        // compare
        bool eq = std::equal(name.first, name.first + name.second, name_.begin(), name_.end());
        if (eq) {
            return true;
        }
        return false;
    }
};

/**
 * @brief Tag value pair
 */
class TagValuePair {
    std::string value_;  //! Value that holds both tag and value
public:
    TagValuePair(const char* begin, const char* end)
        : value_(begin, end)
    {
    }

    TagValuePair(const char* str)
        : value_(str)
    {
    }

    StringT get_value() const {
        return std::make_pair(value_.data(), value_.size());
    }

    bool check(const char* begin, const char* end) const {
        const char* p = begin;
        // skip metric name
        p = skip_space(p, end);
        if (p == end) {
            return false;
        }
        while(*p != ' ') {
            p++;
        }
        p = skip_space(p, end);
        if (p == end) {
            return false;
        }
        // Check tags
        bool error = false;
        while (!error && p < end) {
            const char* tag_start = p;
            const char* tag_end = skip_tag(tag_start, end, &error);
            bool eq = std::equal(tag_start, tag_end, value_.begin(), value_.end());
            if (eq) {
                return true;
            }
            p = skip_space(tag_end, end);
        }
        return false;
    }
};


class IndexQueryResultsIterator {
    CompressedPListConstIterator it_;
    StringPool const* spool_;
public:
    IndexQueryResultsIterator(CompressedPListConstIterator postinglist, StringPool const* spool)
        : it_(postinglist)
        , spool_(spool)
    {
    }

    StringT operator * () const {
        auto id = *it_;
        auto str = spool_->str(id);
        return str;
    }

    IndexQueryResultsIterator& operator ++ () {
        ++it_;
        return *this;
    }

    bool operator == (IndexQueryResultsIterator const& other) const {
        return it_ == other.it_;
    }

    bool operator != (IndexQueryResultsIterator const& other) const {
        return it_ != other.it_;
    }
};

class IndexQueryResults {
    CompressedPList postinglist_;
    StringPool const* spool_;
public:
    IndexQueryResults()
        : spool_(nullptr)
    {}

    IndexQueryResults(CompressedPList&& plist, StringPool const* spool)
        : postinglist_(plist)
        , spool_(spool)
    {
    }

    IndexQueryResults(IndexQueryResults const& other)
        : postinglist_(other.postinglist_)
        , spool_(other.spool_)
    {
    }

    IndexQueryResults& operator = (IndexQueryResults && other) {
        if (this == &other) {
            return *this;
        }
        postinglist_ = std::move(other.postinglist_);
        spool_ = other.spool_;
        return *this;
    }

    IndexQueryResults(IndexQueryResults&& plist)
        : postinglist_(std::move(plist.postinglist_))
        , spool_(plist.spool_)
    {
    }

    template<class Checkable>
    IndexQueryResults filter(std::vector<Checkable> const& values) {
        bool rewrite = false;
        // Check for falce positives
        for (auto it = postinglist_.begin(); it != postinglist_.end(); ++it) {
            auto id = *it;
            auto str = spool_->str(id);
            for (auto const& value: values) {
                if (!value.check(str.first, str.first + str.second)) {
                    rewrite = true;
                    break;
                }
            }
        }
        if (rewrite) {
            // This code only gets triggered when false positives are present
            CompressedPList newplist;
            for (auto it = postinglist_.begin(); it != postinglist_.end(); ++it) {
                auto id = *it;
                auto str = spool_->str(id);
                for (auto const& value: values) {
                    if (value.check(str.first, str.first + str.second)) {
                        newplist.add(id);
                    }
                }
            }
            return IndexQueryResults(std::move(newplist), spool_);
        }
        return *this;
    }

    template<class Checkable>
    IndexQueryResults filter(Checkable const& value) {
        bool rewrite = false;
        // Check for falce positives
        for (auto it = postinglist_.begin(); it != postinglist_.end(); ++it) {
            auto id = *it;
            auto str = spool_->str(id);
            if (!value.check(str.first, str.first + str.second)) {
                rewrite = true;
                break;
            }
        }
        if (rewrite) {
            // This code only gets triggered when false positives are present
            CompressedPList newplist;
            for (auto it = postinglist_.begin(); it != postinglist_.end(); ++it) {
                auto id = *it;
                auto str = spool_->str(id);
                if (value.check(str.first, str.first + str.second)) {
                    newplist.add(id);
                }
            }
            return IndexQueryResults(std::move(newplist), spool_);
        }
        return *this;
    }

    IndexQueryResults intersection(IndexQueryResults const& other) {
        if (spool_ == nullptr) {
            spool_ = other.spool_;
        }
        IndexQueryResults result(postinglist_ & other.postinglist_, spool_);
        return result;
    }

    IndexQueryResults difference(IndexQueryResults const& other) {
        if (spool_ == nullptr) {
            spool_ = other.spool_;
        }
        IndexQueryResults result(postinglist_ ^ other.postinglist_, spool_);
        return result;
    }

    IndexQueryResults join(IndexQueryResults const& other) {
        if (spool_ == nullptr) {
            spool_ = other.spool_;
        }
        IndexQueryResults result(postinglist_ | other.postinglist_, spool_);
        return result;
    }

    size_t cardinality() const {
        return postinglist_.cardinality();
    }

    IndexQueryResultsIterator begin() const {
        return IndexQueryResultsIterator(postinglist_.begin(), spool_);
    }

    IndexQueryResultsIterator end() const {
        return IndexQueryResultsIterator(postinglist_.end(), spool_);
    }
};

struct IndexBase {
    virtual ~IndexBase() = default;
    virtual IndexQueryResults tagvalue_query(TagValuePair const& value) const = 0;
    virtual IndexQueryResults metric_query(MetricName const& value) const = 0;
    virtual std::vector<StringT> list_metric_names() const = 0;
    virtual std::vector<StringT> list_tags(StringT metric) const = 0;
    virtual std::vector<StringT> list_tag_values(StringT metric, StringT tag) const = 0;
};

class IndexQueryNodeBase {
    const char* const name_;

public:

    /**
     * @brief IndexQueryNodeBase c-tor
     * @param name is a static string that contains node name (used for introspection)
     */
    IndexQueryNodeBase(const char* name)
        : name_(name)
    {
    }

    virtual ~IndexQueryNodeBase() = default;

    virtual IndexQueryResults query(const IndexBase&) const = 0;

    const char* get_name() const {
        return name_;
    }
};

/**
 * Extracts only series that have specified tag-value
 * combinations.
 */
struct IncludeTags : IndexQueryNodeBase {
    constexpr static const char* node_name_ = "include-tags";
    MetricName metric_;
    std::vector<TagValuePair> pairs_;

    template<class Iter>
    IncludeTags(MetricName const& metric, Iter begin, Iter end)
        : IndexQueryNodeBase(node_name_)
        , metric_(metric)
        , pairs_(begin, end)
    {
    }

    virtual IndexQueryResults query(IndexBase const&) const;
};

IndexQueryResults IncludeTags::query(IndexBase const& index) const {
    IndexQueryResults results = index.metric_query(metric_);
    for(auto const& tv: pairs_) {
        auto res = index.tagvalue_query(tv);
        results = results.intersection(res);
    }
    return results.filter(metric_).filter(pairs_);
}

/**
 * Extracts only series that have specified tag-value
 * combinations.
 */
struct IncludeIfHasTag : IndexQueryNodeBase {
    constexpr static const char* node_name_ = "include-tags";
    MetricName metric_;
    StringT tagname_;

    IncludeIfHasTag(MetricName const& metric, StringT tag_name)
        : IndexQueryNodeBase(node_name_)
        , metric_(metric)
        , tagname_(tag_name)
    {
    }

    virtual IndexQueryResults query(IndexBase const&) const;
};

IndexQueryResults IncludeIfHasTag::query(IndexBase const& index) const {
    // Query available tag=value pairs first
    std::vector<TagValuePair> pairs;
    auto values = index.list_tag_values(metric_.get_value(), tagname_);
    for (auto val: values) {
        std::stringstream str;
        str << fromstrt(tagname_) << '=' << fromstrt(val);
        auto kv = str.str();
        pairs.emplace_back(kv.c_str());
    }
    IndexQueryResults results = index.metric_query(metric_);
    for(auto const& tv: pairs) {
        auto res = index.tagvalue_query(tv);
        results = results.intersection(res);
    }
    return results.filter(metric_).filter(pairs);
}

/**
 * Extracts only series that doesn't have specified tag-value
 * combinations.
 */
struct ExcludeTags : IndexQueryNodeBase {
    constexpr static const char* node_name_ = "exclude-tags";
    MetricName metric_;
    std::vector<TagValuePair> pairs_;

    template<class Iter>
    ExcludeTags(MetricName const& metric, Iter begin, Iter end)
        : IndexQueryNodeBase(node_name_)
        , metric_(metric)
        , pairs_(begin, end)
    {
    }

    virtual IndexQueryResults query(IndexBase const&) const;
};

IndexQueryResults ExcludeTags::query(IndexBase const& index) const {
    IndexQueryResults results = index.metric_query(metric_);
    for(auto const& tv: pairs_) {
        auto res = index.tagvalue_query(tv);
        results = results.difference(res);
    }
    return results.filter(metric_);
}


struct JoinByTags : IndexQueryNodeBase {
    constexpr static const char* node_name_ = "join-by-tags";
    std::vector<MetricName> metrics_;
    std::vector<TagValuePair> pairs_;

    template<class MIter, class TIter>
    JoinByTags(MIter mbegin, MIter mend, TIter tbegin, TIter tend)
        : IndexQueryNodeBase(node_name_)
        , metrics_(mbegin, mend)
        , pairs_(tbegin, tend)
    {
    }

    virtual IndexQueryResults query(IndexBase const&) const;
};

IndexQueryResults JoinByTags::query(IndexBase const& index) const {
    IndexQueryResults results;
    for(auto const& m: metrics_) {
        auto res = index.metric_query(m);
        results = results.join(res);
    }
    for(auto const& tv: pairs_) {
        auto res = index.tagvalue_query(tv);
        results.difference(res);
    }
    return results.filter(metrics_).filter(pairs_);
}

/**
 * @brief The IndexQuery class
 *
 * Index query is an expression tree composed from
 * tag-value pairs and conditions:
 *
 * - difference
 * - intersection
 * - union
 *
 * E.g.
 *  union(
 *      intersection(tag1=value1, tag2=value2, tag3=value3),
 *      intersection(tag4=value4, tag5=value5)
 *  )
 */
class IndexQuery {
public:
};


/**
 * @brief Split tag=value pair into tag and value
 * @return true on success, false otherwise
 */
static bool split_pair(StringT pair, StringT* outtag, StringT* outval) {
    const char* p = pair.first;
    const char* end = p + pair.second;
    while (*p != '=' && p < end) {
        p++;
    }
    if (p == end) {
        return false;
    }
    *outtag = std::make_pair(pair.first, p - pair.first);
    *outval = std::make_pair(p + 1, pair.second - (p - pair.first + 1));
    return true;
}


class SeriesNameTopology {
    typedef StringTools::L3TableT IndexT;
    IndexT index_;
public:
    SeriesNameTopology()
        : index_(StringTools::create_l3_table(1000))
    {
    }

    void add_name(StringT name) {
        StringT metric = skip_metric_name(name.first, name.first + name.second);
        StringT tags = std::make_pair(name.first + metric.second, name.second - metric.second);
        auto it = index_.find(metric);
        if (it == index_.end()) {
            StringTools::L2TableT tagtable = StringTools::create_l2_table(1024);
            index_[metric] = std::move(tagtable);
            it = index_.find(metric);
        }
        // Iterate through tags
        const char* p = tags.first;
        const char* end = p + tags.second;
        p = skip_space(p, end);
        if (p == end) {
            return;
        }
        // Check tags
        bool error = false;
        while (!error && p < end) {
            const char* tag_start = p;
            const char* tag_end = skip_tag(tag_start, end, &error);
            auto tagstr = std::make_pair(tag_start, tag_end - tag_start);
            StringT tag;
            StringT val;
            if (!split_pair(tagstr, &tag, &val)) {
                error = true;
            }
            StringTools::L2TableT& tagtable = it->second;
            auto tagit = tagtable.find(tag);
            if (tagit == tagtable.end()) {
                auto valtab = StringTools::create_set(1024);
                tagtable[tag] = std::move(valtab);
                tagit = tagtable.find(tag);
            }
            StringTools::SetT& valueset = tagit->second;
            valueset.insert(val);
            // next
            p = skip_space(tag_end, end);
        }
    }

    std::vector<StringT> list_metric_names() const {
        std::vector<StringT> res;
        std::transform(index_.begin(), index_.end(), std::back_inserter(res),
                       [](std::pair<StringT, StringTools::L2TableT> const& v) {
                            return v.first;
                       });
        return res;
    }

    std::vector<StringT> list_tags(StringT metric) const {
        std::vector<StringT> res;
        auto it = index_.find(metric);
        if (it == index_.end()) {
            return res;
        }
        std::transform(it->second.begin(), it->second.end(), std::back_inserter(res),
                       [](std::pair<StringT, StringTools::SetT> const& v) {
                            return v.first;
                       });
        return res;
    }

    std::vector<StringT> list_tag_values(StringT metric, StringT tag) const {
        std::vector<StringT> res;
        auto it = index_.find(metric);
        if (it == index_.end()) {
            return res;
        }
        auto vit = it->second.find(tag);
        if (vit == it->second.end()) {
            return res;
        }
        const auto& set = vit->second;
        std::copy(set.begin(), set.end(), std::back_inserter(res));
        return res;
    }
};

class Index : public IndexBase {
    StringPool pool_;
    StringTools::TableT table_;
    CMSketch metrics_names_;
    CMSketch tagvalue_pairs_;
    SeriesNameTopology topology_;
public:
    Index()
        : table_(StringTools::create_table(100000))
        , metrics_names_(3, 1024)
        , tagvalue_pairs_(3, 1024)
    {
    }

    SeriesNameTopology const& get_topology() const {
        return topology_;
    }

    size_t cardinality() const {
        return table_.size();
    }

    size_t memory_use() const {
        // TODO: use counting allocator for table_ to provide memory stats
        size_t sm = metrics_names_.get_size_in_bytes();
        size_t st = tagvalue_pairs_.get_size_in_bytes();
        size_t sp = pool_.mem_used();
        return sm + st + sp;
    }

    size_t index_memory_use() const {
        // TODO: use counting allocator for table_ to provide memory stats
        size_t sm = metrics_names_.get_size_in_bytes();
        size_t st = tagvalue_pairs_.get_size_in_bytes();
        return sm + st;
    }

    size_t pool_memory_use() const {
        return pool_.mem_used();
    }

    aku_Status append(const char* begin, const char* end) {
        // Parse string value and sort tags alphabetically
        const char* tags_begin;
        const char* tags_end;
        char buffer[0x1000];
        auto status = SeriesParser::to_normal_form(begin, end, buffer, buffer + 0x1000, &tags_begin, &tags_end);
        if (status != AKU_SUCCESS) {
            std::string line(begin, end);
            std::cout << "error: " << status << std::endl;
            std::cout << "line: " << line << std::endl;
            return AKU_EBAD_ARG;
        }
        // Check if name is already been added
        auto name = std::make_pair((const char*)buffer, tags_end - buffer);
        if (table_.count(name) == 0) {
            // insert value
            auto id = pool_.add(buffer, tags_end);
            if (id == 0) {
                std::string line(begin, end);
                std::cout << "can't add string \"" << line << "\"" << std::endl;
                return AKU_EBAD_DATA;
            }
            write_tags(tags_begin, tags_end, &tagvalue_pairs_, id);
            name = pool_.str(id);  // name now have the same lifetime as pool
            table_[name] = id;
            auto mname = skip_metric_name(buffer, tags_begin);
            if (mname.second == 0) {
                return AKU_EBAD_DATA;
            }
            auto mhash = StringTools::hash(mname);
            metrics_names_.add(mhash, id);
            // update topology
            topology_.add_name(name);
        }
        return AKU_SUCCESS;
    }

    virtual IndexQueryResults tagvalue_query(const TagValuePair &value) const {
        auto hash = StringTools::hash(value.get_value());
        auto post = tagvalue_pairs_.extract(hash);
        return IndexQueryResults(std::move(post), &pool_);
    }

    virtual IndexQueryResults metric_query(const MetricName &value) const {
        auto hash = StringTools::hash(value.get_value());
        auto post = metrics_names_.extract(hash);
        return IndexQueryResults(std::move(post), &pool_);
    }

    virtual std::vector<StringT> list_metric_names() const {
        return topology_.list_metric_names();
    }

    virtual std::vector<StringT> list_tags(StringT metric) const {
        return topology_.list_tags(metric);
    }

    virtual std::vector<StringT> list_tag_values(StringT metric, StringT tag) const {
        return topology_.list_tag_values(metric, tag);
    }
};

int main(int argc, char *argv[])
{
    Index index;
    //for (auto line: sample_lines) {
    for (std::string line; std::getline(std::cin, line);) {
        index.append(line.data(), line.data() + line.size());
    }
    std::cout << "number of unique series: " << index.cardinality() << std::endl;
    std::cout << "memory use: " << index.memory_use() << " bytes" << std::endl;
    std::cout << "string pool size: " << index.pool_memory_use() << " bytes" << std::endl;
    std::cout << "index size: " << index.index_memory_use() << " bytes" << std::endl;

    // Try to extract by tag combination
    const char* tag_host = "host=192.168.160.245";
    const char* tag_inst = "instance-type=m3.large";
    std::vector<TagValuePair> tgv;
    tgv.emplace_back(tag_host);
    tgv.emplace_back(tag_inst);
    MetricName mname("cpu.user");
    IncludeTags tags(mname, tgv.begin(), tgv.end());
    PerfTimer tm2;
    auto results = tags.query(index);
    double elapsed = tm2.elapsed();
    std::cout << "Tag combination extracted, time: " << (elapsed*1000000.0) << " usec" << std::endl;
    for (auto it = results.begin(); it != results.end(); ++it) {
        auto sname = *it;
        std::cout << "Name found: " << std::string(sname.first, sname.first + sname.second) << std::endl;
    }
    auto values = index.get_topology().list_tag_values(tostrt("cpu.user"), tostrt("host"));
    std::cout << "found " << values.size() << " values" << std::endl;
    return 0;
}

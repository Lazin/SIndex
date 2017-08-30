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
    StringT str(u64 bits);

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

StringT StringPool::str(u64 bits) {
    u64 ix     = bits / MAX_BIN_SIZE;
    u64 offset = bits % MAX_BIN_SIZE;
    std::lock_guard<std::mutex> guard(pool_mutex);
    if (ix < pool.size()) {
        std::vector<char>* bin = &pool.at(ix);
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

    //! Inverted table type (id to string mapping)
    typedef std::unordered_map<u64, StringT> InvT;

    static TableT create_table(size_t size);

static SetT create_set(size_t size);
};

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

//             //
//  CM-sketch  //
//             //

class CMSketch {

};

//               //
//  PostingList  //
//               //

/**
 * @brief The PostingList class can be used to read and write posting lists
 */
class PostingList {

};



int main(int argc, char *argv[])
{
    StringPool pool;
    StringTools::TableT table = StringTools::create_table(100000);
    StringTools::InvT inv_tab;
    char buffer[0x1000];
    for (std::string line; std::getline(std::cin, line);) {
        const char* tags_begin;
        const char* tags_end;
        auto status = SeriesParser::to_normal_form(line.data(), line.data() + line.size(), buffer, buffer + 0x1000, &tags_begin, &tags_end);
        if (status != AKU_SUCCESS) {
            std::cout << "error: " << status << std::endl;
            std::cout << "line: " << line << std::endl;
            std::abort();
        }
        auto name = std::make_pair((const char*)buffer, tags_end - buffer);
        if (table.count(name) == 0) {
            auto id = pool.add(buffer, tags_end);
            if (id == 0) {
                std::cout << "can't add string \"" << line << "\"" << std::endl;
                std::abort();
            }
            name = pool.str(id);  // name now have the same lifetime as pool
            table[name] = id;
        }
    }
    std::cout << "number of unique series: " << table.size() << std::endl;
    std::cout << "string pool size: " << pool.size() << " lines, " << pool.mem_used() << " bytes" << std::endl;
    return 0;
}

/**
 * @file MemoryMgmt.h
 * @date 2017-12-05
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef MEMORYMGMT_H
#define MEMORYMGMT_H 

#include <stdlib.h>
#include <limits.h>

#include <map>
#include <mutex>


typedef struct MemoryEntry_s {
    char            filename[PATH_MAX];
    char            funcname[PATH_MAX];
    int             line;
    unsigned long   size;
} MemoryEntry;

#define SMALLOC_(obj, type, size, filename, func, line)                                 \
    do {                                                                                \
        void* ptr = malloc(size);                                                       \
        if (ptr == NULL) {                                                              \
            obj = NULL;                                                                 \
        } else {                                                                        \
            obj = (type*) ptr;                                                          \
            MemoryMgmt::insertEntry(filename, func, line, size, ptr);                   \
        }                                                                               \
    } while (0)

#define SMALLOC_NOLOG_(obj, type, size)                                                 \
    do {                                                                                \
        void* ptr = malloc(size);                                                       \
        if (ptr == NULL) {                                                              \
            obj = NULL;                                                                 \
        } else {                                                                        \
            obj = (type*) ptr;                                                          \
        }                                                                               \
    } while (0)

#define SFREE_(obj)                                                                     \
    do {                                                                                \
        free(obj);                                                                      \
        MemoryMgmt::removeEntry((void*)obj);                                            \
    } while (0)

#define SFREE_NOLOG_(obj)                                                               \
    do {                                                                                \
        free(obj);                                                                      \
    } while (0)

#define SNEW_(obj, type, filename, func, line, ...)                                     \
    do {                                                                                \
        int size = sizeof( type );                                                      \
        void* ptr = malloc(size);                                                       \
        if (ptr == NULL) {                                                              \
            obj = NULL;                                                                 \
        } else {                                                                        \
            obj = new ((type*)ptr) type(__VA_ARGS__);                                   \
            MemoryMgmt::insertEntry(filename, func, line, size, ptr);                   \
        }                                                                               \
    } while (0)

#define SNEW_NOLOG_(obj, type, ...)                                                     \
    do {                                                                                \
        int size = sizeof( type );                                                      \
        void* ptr = malloc(size);                                                       \
        if (ptr == NULL) {                                                              \
            obj = NULL;                                                                 \
        } else {                                                                        \
            obj = new ((type*)ptr) type(__VA_ARGS__);                                   \
        }                                                                               \
    } while (0)

#define SDELETE_(obj)                                                                   \
    do {                                                                                \
        delete obj;                                                                     \
        MemoryMgmt::removeEntry((void*)obj);                                            \
    } while (0)

#define SDELETE_NOLOG_(obj)                                                             \
    do {                                                                                \
        delete obj;                                                                     \
    } while (0)

/* NOTE: we can use __func__ macro because we use C++11 */
#ifdef DEBUG_MODE
#define SMALLOC(obj, type, size)    SMALLOC_(obj, type, size, __FILE__, __func__, __LINE__) 
#define SNEW(obj, type, args...)    SNEW_(obj, type, __FILE__, __func__, __LINE__, ##args )
#define SDELETE(obj)                SDELETE_(obj)
#define SFREE(obj)                  SFREE_(obj)
#else
#define SMALLOC(obj, type, size)    SMALLOC_NOLOG_(obj, type, size)
#define SNEW(obj, type, args...)    SNEW_NOLOG_(obj, type, ##args )
#define SDELETE(obj)                SDELETE_NOLOG_(obj)
#define SFREE(obj)                  SFREE_NOLOG_(obj)
#endif

class MemoryMgmt {
public: 
    MemoryMgmt() {}
    virtual ~MemoryMgmt() {}

    static void init();
    static void insertEntry(const char* filename, const char* funcname, int line,
        unsigned long size, void* ptr);
    static void removeEntry(void* ptr);
    static void dump();

private:
    static std::map<void*, MemoryEntry>     entryMap;
    static std::mutex                       entryMutex;
    static uint64_t                         usedMemTotalSize;
};
#endif /* MEMORYMGMT_H */

#ifndef IMPORT_HELPER_H_INCLUDED
#define IMPORT_HELPER_H_INCLUDED

#define BOOST_PYTHON_STATIC_LIB
// The following defines did not help
//#define BOOST_PYTHON_ENABLE_STDCALL
//#define BOOST_PYTHON_ENABLE_CDECL
#include <boost/python.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/shared_mutex.hpp>
#include "LibPyNDI.h"
#include <stdio.h>

class ScopedGILRelease
{
public:
    inline ScopedGILRelease()
    {
        m_thread_state = PyEval_SaveThread();
    }

    inline ~ScopedGILRelease()
    {
        PyEval_RestoreThread(m_thread_state);
        m_thread_state = NULL;
    }

private:
    PyThreadState * m_thread_state;
};


class ScopePrinter
{
public:
	inline ScopePrinter()
    {
        printf("(");
    }

    inline ~ScopePrinter()
    {
        printf(")");
    }
};

//typedef boost::shared_mutex Lock;
//typedef boost::unique_lock< Lock > UniqueLock;
//typedef boost::shared_lock< Lock > SharedLock;

typedef boost::mutex Lock;
typedef boost::lock_guard< Lock > UniqueLock;

extern Lock LibNDILock;

#define SCOPED_GIL_RELEASE ScopedGILRelease scoped;
#define LIB_NDI_THREAD_LOCK UniqueLock libndilock(LibNDILock);
#define LIB_NDI_SCOPE_PRINTER ScopePrinter scopeprinter;


class LockingScopePrinter
{
public:
	UniqueLock *libndilock;
    inline LockingScopePrinter()
    {
        printf("(");
		this->libndilock = new UniqueLock(LibNDILock);
		printf("<");
    }

    inline ~LockingScopePrinter()
    {
        printf(">");
		delete this->libndilock;
		printf(")");
    }
};

#define LIB_NDI_LOCKING_SCOPE_PRINTER LockingScopePrinter scopeprinter;

//#define LIB_NDI_SCOPE_LOCK	LIB_NDI_THREAD_LOCK; SCOPED_GIL_RELEASE;
//#define LIB_NDI_SCOPE_LOCK	LIB_NDI_THREAD_LOCK;
//#define LIB_NDI_SCOPE_LOCK	LIB_NDI_SCOPE_PRINTER;
//#define LIB_NDI_SCOPE_LOCK	LIB_NDI_LOCKING_SCOPE_PRINTER;
#define LIB_NDI_SCOPE_LOCK	LIB_NDI_THREAD_LOCK;


// boost does not allow wrapping __stdcall, so we need intermediate functions   
#define WRAP_INT_FUNCTION_STRING(f) \
	int _##f (std::string s) { \
		LIB_NDI_SCOPE_LOCK; \
		return f((char*)s.c_str()); \
	}
#define WRAP_INT_FUNCTION_INT(f) \
	int _##f (int i) { \
		LIB_NDI_SCOPE_LOCK; \
		return f(i); \
	}
#define WRAP_INT_FUNCTION_UINT(f) \
	int _##f (unsigned int i) { \
		LIB_NDI_SCOPE_LOCK; \
		return f(i); \
	}
#define WRAP_INT_FUNCTION_VOID(f) \
	int _##f (void) { \
		LIB_NDI_SCOPE_LOCK; \
		return f(); \
	}

#define FUNCTION_DEF(f) boost::python::def(#f, _##f)

#endif // !IMPORT_HELPER_H_INCLUDED
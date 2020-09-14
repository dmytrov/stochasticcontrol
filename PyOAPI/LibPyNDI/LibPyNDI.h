// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the LIBPYNDI_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// LIBPYNDI_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifndef LIB_PY_NDI_H_INCLUDED
#define LIB_PY_NDI_H_INCLUDED

#ifdef LIBPYNDI_EXPORTS
#define LIBPYNDI_API __declspec(dllexport)
#else
#define LIBPYNDI_API __declspec(dllimport)
#endif

// This class is exported from the LibPyNDI.dll
class LIBPYNDI_API CLibPyNDI {
public:
	CLibPyNDI(void);
	// TODO: add your methods here.
};

extern LIBPYNDI_API int nLibPyNDI;

LIBPYNDI_API int fnLibPyNDI(void);

#endif // !LIB_PY_NDI_H_INCLUDED
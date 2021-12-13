PUSHPOP_C = '''
/*
 * TAPENADE Automatic Differentiation Engine
 * Copyright (C) 1999-2021 Inria
 * See the LICENSE.md file in the project root for more information.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* The size of a BLOCK in characters. Suggested 16384. Should try 2^16=65536 */
#define ONE_BLOCK_SIZE 65536

/* The main stack is a double-chain of DoubleChainedBlock objects.
 * Each DoubleChainedBlock holds an array[ONE_BLOCK_SIZE] of char. */
typedef struct _DoubleChainedBlock{
  unsigned int rank ;
  struct _DoubleChainedBlock *prev ;
  char                       *contents ;
  struct _DoubleChainedBlock *next ;
} DoubleChainedBlock ;

char initContents[ONE_BLOCK_SIZE] = {'\0'} ;
DoubleChainedBlock initBlock = {0,NULL,initContents,NULL} ;
static DoubleChainedBlock *curStack = &initBlock ;
static char               *curStackTop = initContents ;

static unsigned long int maintraffic = 0 ;

void setCurLocation(unsigned long int location) {
  unsigned int targetRank = (unsigned int)location/ONE_BLOCK_SIZE ;
  unsigned int targetOffset = (unsigned int)location%ONE_BLOCK_SIZE ;
  if (targetRank>curStack->rank)
    while (targetRank>curStack->rank) curStack = curStack->next ;
  else if (targetRank<curStack->rank)
    while (targetRank<curStack->rank) curStack = curStack->prev ;
  curStackTop = curStack->contents + targetOffset ;
}

unsigned long int getCurLocation() {
  return (curStackTop-curStack->contents)+curStack->rank*ONE_BLOCK_SIZE ;
}

void showLocation(unsigned long int location) {
  printf("%1i.%05i", (unsigned int)location/ONE_BLOCK_SIZE, (unsigned int)location%ONE_BLOCK_SIZE) ;
}

/*************** REPEATED ACCESS MECHANISM *********************/

typedef struct _StackRepeatCell {
  int hasBackPop ;
  unsigned long int backPop ;
  unsigned long int resume ;
  unsigned long int freePush ;
  struct _StackRepeatCell *previous ;
} StackRepeatCell ;

StackRepeatCell *stackRepeatTop = NULL ;

void showStackRepeatsRec(StackRepeatCell *inRepeatStack) {
  if (inRepeatStack->previous) {showStackRepeatsRec(inRepeatStack->previous) ; printf(" ; ") ;}
  printf("<") ;
  if (inRepeatStack->hasBackPop) showLocation(inRepeatStack->backPop) ;
  printf("|") ;
  showLocation(inRepeatStack->resume) ;
  printf("|") ;
  showLocation(inRepeatStack->freePush) ;
  printf(">") ;
}

void showStackRepeats() {
  showStackRepeatsRec(stackRepeatTop) ;
}

void showStack() {
  DoubleChainedBlock *inStack = &initBlock ;
  int i ;
  while (inStack) {
    printf("[%1i] ",inStack->rank) ;
    for (i=0 ; i<ONE_BLOCK_SIZE ; ++i) {
      if (i!=0 && i%4==0) printf(".") ;
      if (inStack==curStack && &(inStack->contents[i])==curStackTop) printf(" | ") ;
      printf("%02x",(unsigned char)inStack->contents[i]) ;
    }
    inStack = inStack->next ;
    if (inStack) printf("\n        ") ;
  }
  printf("\n        REPEATS:") ;
  if (stackRepeatTop)
    showStackRepeats() ;
  else
    printf(" none!") ;
  printf("\n") ;
}

void showStackSize(int i4i, int i8i, int r4i, int r8i, int c8i, int c16i, int s1i, int biti, int ptri, int pos) {
  printf(" --%5i--> <",pos) ;
  showLocation(getCurLocation()) ;
  printf(">%1i.%1i.%1i.%1i.%1i.%1i.%1i.%1i.%1i\n",i4i, i8i, r4i, r8i, c8i, c16i, s1i, biti, ptri) ;
}

void adStack_showPeakSize() {
  DoubleChainedBlock *inStack = &initBlock ;
  int i = 0 ;
  while (inStack) {
    inStack = inStack->next ;
    ++i ;
  }
  printf("Peak stack size (%1i blocks): %1llu bytes\n",
         i, ((long long int)i)*((long long int)ONE_BLOCK_SIZE)) ;
}

void showTotalTraffic(unsigned long long int localtraffic) {
  printf("Total pushed traffic %1llu bytes\n", maintraffic+localtraffic) ;
}

/** If we are in a protected, read-only section, memorize location as "backPop"
 * and go to the "freePush" location */
void checkPushInReadOnly() {
  if (stackRepeatTop) {
    unsigned long int current = getCurLocation() ;
    if (current<stackRepeatTop->freePush) {
      stackRepeatTop->hasBackPop = 1 ;
      stackRepeatTop->backPop = current ;
      setCurLocation(stackRepeatTop->freePush) ;
/*       printf(" FREEPUSH(") ;                   //Trace */
/*       showLocation(stackRepeatTop->backPop) ;  //Trace */
/*       printf("=>") ;                           //Trace */
/*       showLocation(stackRepeatTop->freePush) ; //Trace */
/*       printf(")") ;                            //Trace */
    }
  }
}

/** If current location is the "freePush" location,
 * go back to its "backPop" location, which is in a protected, read-only section */
void checkPopToReadOnly() {
  if (stackRepeatTop && stackRepeatTop->hasBackPop) {
    unsigned long int current = getCurLocation() ;
    if (current==stackRepeatTop->freePush) {
      setCurLocation(stackRepeatTop->backPop) ;
      stackRepeatTop->hasBackPop = 0 ;
/*       printf(" BACKPOP(") ;                    //Trace */
/*       showLocation(stackRepeatTop->freePush) ; //Trace */
/*       printf("=>") ;                           //Trace */
/*       showLocation(stackRepeatTop->backPop) ;  //Trace */
/*       printf(")") ;                            //Trace */
    }
  }
}

// A global for communication from startStackRepeat1() to startStackRepeat2():
StackRepeatCell *newRepeatCell = NULL ;

void startStackRepeat1() {
  // Create (push) a new "stack" repeat level:
  newRepeatCell = (StackRepeatCell *)malloc(sizeof(StackRepeatCell)) ;
  newRepeatCell->previous = stackRepeatTop ;
  newRepeatCell->hasBackPop = 0 ;
  // Store current location as the "resume" location:
  unsigned long int current = getCurLocation() ;
  newRepeatCell->resume = current ;
  // Move to the "freePush" location if there is one:
  if (stackRepeatTop && current<stackRepeatTop->freePush)
    setCurLocation(stackRepeatTop->freePush) ;
}

void startStackRepeat2() {
  // Store current stack location as the "freePush" location:
  newRepeatCell->freePush = getCurLocation() ;
  // Reset current location to stored "resume" location:
  setCurLocation(newRepeatCell->resume) ;
  // Make this new repeat level the current repeat level:
  stackRepeatTop = newRepeatCell ;
/*   printf("\n+Rep ") ; showStackRepeats() ; printf("\n") ; //Trace */
}

void resetStackRepeat1() {
/*   printf("\n>Rep ") ; showStackRepeats() ; printf("\n") ; //Trace */
  // If we are in a nested checkpoint, force exit from it:
  if (stackRepeatTop->hasBackPop) {
    //setCurLocation(stackRepeatTop->backPop) ; //correct but useless code
    stackRepeatTop->hasBackPop = 0 ;
  }
  // Go to repeat location of current repeat level
  setCurLocation(stackRepeatTop->freePush) ;
}

void resetStackRepeat2() {
  // Reset current location to "ResumeLocation":
  setCurLocation(stackRepeatTop->resume) ;
}

void endStackRepeat() {
/*   printf("\n-Rep ") ; showStackRepeats() ; printf("\n") ; //Trace */
  // If we are in a nested checkpoint, go back to its "backPop" (read-only) location:
  if (stackRepeatTop->hasBackPop) {
    setCurLocation(stackRepeatTop->backPop) ;
    //stackRepeatTop->hasBackPop = 0 ; //correct but useless code
  }
  // Remove (pop) top "stack" repeat level:
  StackRepeatCell *oldRepeatCell = stackRepeatTop ;
  stackRepeatTop = stackRepeatTop->previous ;
  free(oldRepeatCell) ;
  // current location may have moved back ; check if we must move further back:
  checkPopToReadOnly() ;
}

/******************* PUSH/POP MECHANISM *******************/

/* PUSHes "nbChars" consecutive chars from a location starting at address "x".
 * Checks that there is enough space left to hold "nbChars" chars.
 * Otherwise, allocates the necessary space. */
void pushNArray(char *x, unsigned int nbChars, int checkReadOnly) {
  if (checkReadOnly) checkPushInReadOnly() ;
  if (checkReadOnly) maintraffic += nbChars ;
/* unsigned long int lfrom = getCurLocation() ; //Trace */
  unsigned int nbmax = ONE_BLOCK_SIZE-(curStackTop-(curStack->contents)) ;
  if (nbChars <= nbmax) {
    memcpy(curStackTop,x,nbChars) ;
    curStackTop+=nbChars ;
  } else {
    char *inx = x+(nbChars-nbmax) ;
    if (nbmax>0) memcpy(curStackTop,inx,nbmax) ;
    while (inx>x) {
      if (curStack->next)
        curStack = curStack->next ;
      else {
        /* Create new block: */
	DoubleChainedBlock *newStack ;
	char *contents = (char *)malloc(ONE_BLOCK_SIZE*sizeof(char)) ;
	newStack = (DoubleChainedBlock*)malloc(sizeof(DoubleChainedBlock)) ;
	if ((contents == NULL) || (newStack == NULL)) {
	  DoubleChainedBlock *stack = curStack ;
	  int nbBlocks = (stack?-1:0) ;
	  while(stack) {
	      stack = stack->prev ;
	      nbBlocks++ ;
	  }
	  printf("Out of memory (allocated %i blocks of %i bytes)\n",
		 nbBlocks, ONE_BLOCK_SIZE) ;
          exit(0);
	}
        curStack->next = newStack ;
	newStack->prev = curStack ;
        newStack->rank = curStack->rank + 1 ;
	newStack->next = NULL ;
	newStack->contents = contents ;
	curStack = newStack ;
        /* new block created! */
      }
      inx -= ONE_BLOCK_SIZE ;
      if(inx>x)
        memcpy(curStack->contents,inx,ONE_BLOCK_SIZE) ;
      else {
        unsigned int nbhead = (inx-x)+ONE_BLOCK_SIZE ;
        curStackTop = curStack->contents ;
        memcpy(curStackTop,x,nbhead) ;
        curStackTop += nbhead ;
      }
    }
  }
/* unsigned long int lto = getCurLocation() ; //Trace */
/* printf("pushNArray(") ;                    //Trace */
/* showLocation(lfrom) ;                      //Trace */
/* printf("=>") ;                             //Trace */
/* showLocation(lto) ;                        //Trace */
/* printf(")") ;                              //Trace */
}

/* POPs "nbChars" consecutive chars to a location starting at address "x".
 * Checks that there is enough data to fill "nbChars" chars.
 * Otherwise, pops as many blocks as necessary. */
void popNArray(char *x, unsigned int nbChars, int checkReadOnly) {
/* unsigned long int lfrom = getCurLocation() ; //Trace */
  unsigned int nbmax = curStackTop-(curStack->contents) ;
  if (nbChars <= nbmax) {
    curStackTop-=nbChars ;
    memcpy(x,curStackTop,nbChars);
  } else {
    char *tlx = x+nbChars ;
    if (nbmax>0) memcpy(x,curStack->contents,nbmax) ;
    x+=nbmax ;
    while (x<tlx) {
      curStack = curStack->prev ;
      if (curStack==NULL) printf("Popping from an empty stack!!!\n") ;
      if (x+ONE_BLOCK_SIZE<tlx) {
	memcpy(x,curStack->contents,ONE_BLOCK_SIZE) ;
	x += ONE_BLOCK_SIZE ;
      } else {
	unsigned int nbtail = tlx-x ;
	curStackTop = (curStack->contents)+ONE_BLOCK_SIZE-nbtail ;
	memcpy(x,curStackTop,nbtail) ;
	x = tlx ;
      }
    }
  }
/* unsigned long int lto = getCurLocation() ; //Trace */
/* printf("popNArray(") ;                     //Trace */
/* showLocation(lfrom) ;                      //Trace */
/* printf("=>") ;                             //Trace */
/* showLocation(lto) ;                        //Trace */
/* printf(")") ;                              //Trace */
  if (checkReadOnly) checkPopToReadOnly() ;
}

typedef struct {float r,i;} ccmplx ;
typedef struct {double dr, di;} cdcmplx ;

void pushInteger4Array(int *x, int n) {
  pushNArray((char *)x,(unsigned int)(n*4), 1) ;
}

void popInteger4Array(int *x, int n) {
  popNArray((char *)x,(unsigned int)(n*4), 1) ;
}

void pushInteger8Array(long *x, int n) {
  pushNArray((char *)x,(unsigned int)(n*8), 1) ;
}

void popInteger8Array(long *x, int n) {
  popNArray((char *)x,(unsigned int)(n*8), 1) ;
}

void pushReal4Array(float *x, int n) {
  pushNArray((char *)x,(unsigned int)(n*4), 1) ;
}

void popReal4Array(float *x, int n) {
  popNArray((char *)x,(unsigned int)(n*4), 1) ;
}

void pushReal8Array(double *x, int n) {
  pushNArray((char *)x,(unsigned int)(n*8), 1) ;
}

void popReal8Array(double *x, int n) {
  popNArray((char *)x,(unsigned int)(n*8), 1) ;
}

void pushComplex8Array(ccmplx *x, int n) {
  pushNArray((char *)x,(unsigned int)(n*8), 1) ;
}

void popComplex8Array(ccmplx *x, int n) {
  popNArray((char *)x,(unsigned int)(n*8), 1) ;
}

void pushComplex16Array(cdcmplx *x, int n) {
  pushNArray((char *)x,(unsigned int)(n*16), 1) ;
}

void popComplex16Array(cdcmplx *x, int n) {
  popNArray((char *)x,(unsigned int)(n*16), 1) ;
}

void pushCharacterArray(char *x, int n) {
  pushNArray(x,(unsigned int)n, 1) ;
}

void popCharacterArray(char *x, int n) {
  popNArray(x,(unsigned int)n, 1) ;
}

/* ********* Useful only for testpushpop.f90. Should go away! ********* */

void showpushpopsequence_(int *op, int *index, int* nbobjects, int* sorts, int* sizes) {
  char *prefix = "" ;
  if (*op==1) prefix = "+" ;
  else if (*op==-1) prefix = "-" ;
  else if (*op==2) prefix = "+s" ;
  else if (*op==-2) prefix = "-s" ;
  else if (*op==-3) prefix = "Ls" ;
  printf("%s%02i", prefix, *index) ;
  // Comment the rest for compact display:
  printf(":") ;
  int i ;
  for (i=0 ; i<*nbobjects ; ++i) {
    switch (sorts[i]) {
    case 1:
      printf(" I4") ;
      break ;
    case 2:
      printf(" I8") ;
      break ;
    case 3:
      printf(" R4") ;
      break ;
    case 4:
      printf(" R8") ;
      break ;
    case 5:
      printf(" C8") ;
      break ;
    case 6:
      printf(" C16") ;
      break ;
    case 7:
      printf(" char") ;
      break ;
    case 8:
      printf(" bit") ;
      break ;
    case 9:
      printf(" PTR") ;
      break ;
    }
    if (sizes[i]!=0) printf("[%1i]",sizes[i]) ;
  }
}

/****************** INTERFACE CALLED FROM FORTRAN *******************/

void showstack_() {
  showStack() ;
}

void showstacksize_(int *i4i, int *i8i, int *r4i, int *r8i, int *c8i, int *c16i, int *s1i, int *biti, int *ptri, int *pos) {
  showStackSize(*i4i,*i8i,*r4i,*r8i,*c8i,*c16i,*s1i,*biti,*ptri, *pos) ;
}

void adstack_showpeaksize_() {
  adStack_showPeakSize() ;
}

void adstack_showpeaksize__() {
  adStack_showPeakSize() ;
}

void showtotaltraffic_(unsigned long long int *traffic) {
  showTotalTraffic(*traffic) ;
}

void startstackrepeat1_() {
  startStackRepeat1() ;
}

void startstackrepeat2_() {
  startStackRepeat2() ;
}

void resetstackrepeat1_() {
  resetStackRepeat1() ;
}

void resetstackrepeat2_() {
  resetStackRepeat2() ;
}

void endstackrepeat_() {
  endStackRepeat() ;
}

void pushnarray_(char *x, unsigned int *nbChars, int *checkReadOnly) {
  pushNArray(x, *nbChars, *checkReadOnly) ;
}

void popnarray_(char *x, unsigned int *nbChars, int *checkReadOnly) {
  popNArray(x, *nbChars, *checkReadOnly) ;
}

void pushinteger4array_(int *ii, int *ll) {
  pushInteger4Array(ii, *ll) ;
}

void popinteger4array_(int *ii, int *ll) {
  popInteger4Array(ii, *ll) ;
}

void pushinteger8array_(long *ii, int *ll) {
  pushInteger8Array(ii, *ll) ;
}

void popinteger8array_(long *ii, int *ll) {
  popInteger8Array(ii, *ll) ;
}

void pushreal4array_(float *ii, int *ll) {
  pushReal4Array(ii, *ll) ;
}

void popreal4array_(float *ii, int *ll) {
  popReal4Array(ii, *ll) ;
}

void pushreal8array_(double *ii, int *ll) {
  pushReal8Array(ii, *ll) ;
}

void popreal8array_(double *ii, int *ll) {
  popReal8Array(ii, *ll) ;
}

void pushcomplex8array_(ccmplx *ii, int *ll) {
  pushComplex8Array(ii, *ll) ;
}

void popcomplex8array_(ccmplx *ii, int *ll) {
  popComplex8Array(ii, *ll) ;
}

void pushcomplex16array_(cdcmplx *ii, int *ll) {
  pushComplex16Array(ii, *ll) ;
}

void popcomplex16array_(cdcmplx *ii, int *ll) {
  popComplex16Array(ii, *ll) ;
}

void pushcharacterarray_(char *ii, int *ll) {
  pushCharacterArray(ii, *ll) ;
}

void popcharacterarray_(char *ii, int *ll) {
  popCharacterArray(ii, *ll) ;
}

void pushbooleanarray_(char *x, unsigned int *n) {
  pushNArray(x,(*n*4), 1) ;
}

void popbooleanarray_(char *x, unsigned int *n) {
  popNArray(x,(*n*4), 1) ;
}




///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


/************ MEASUREMENT OF PUSH TRAFFIC *************/

static long int bufferTraffic = 0 ;

void addBufferTraffic(int n) {
  bufferTraffic += n ;
}

void adStack_showTraffic() {
  showTotalTraffic(bufferTraffic) ;
}

/************************** integer*4 ************************/
/* The buffer array for I4. Suggested size 512 */
#define I4BUFSIZE 512
static int adi4buf[I4BUFSIZE] ;
static int adi4ibuf = 0 ;

void pushInteger4(int x) {
  addBufferTraffic(4) ;
  adi4buf[adi4ibuf] = x ;
  if (adi4ibuf>=I4BUFSIZE-1) {
    pushNArray((char *)adi4buf, I4BUFSIZE*4, 1) ;
    addBufferTraffic(-I4BUFSIZE*4) ;
    adi4ibuf = 0 ;
  } else
    ++adi4ibuf ;
}

void popInteger4(int *x) {
  if (adi4ibuf<=0) {
    popNArray((char *)adi4buf, I4BUFSIZE*4, 1) ;
    adi4ibuf = I4BUFSIZE-1 ;
  } else
    --adi4ibuf ;
  *x = adi4buf[adi4ibuf] ;
}

/************************** integer*8 ************************/
/* The buffer array for I8. Suggested size 512 */
#define I8BUFSIZE 512
static long adi8buf[I8BUFSIZE] ;
static int adi8ibuf = 0 ;

void pushInteger8(long x) {
  addBufferTraffic(8) ;
  adi8buf[adi8ibuf] = x ;
  if (adi8ibuf>=I8BUFSIZE-1) {
    pushNArray((char *)adi8buf, I8BUFSIZE*8, 1) ;
    addBufferTraffic(-I8BUFSIZE*8) ;
    adi8ibuf = 0 ;
  } else
    ++adi8ibuf ;
}

void popInteger8(long *x) {
  if (adi8ibuf<=0) {
    popNArray((char *)adi8buf, I8BUFSIZE*8, 1) ;
    adi8ibuf = I8BUFSIZE-1 ;
  } else
    --adi8ibuf ;
  *x = adi8buf[adi8ibuf] ;
}

/************************** real*4 ************************/
// The buffer array for R4. Suggested size 512
#define R4BUFSIZE 512
static float adr4buf[R4BUFSIZE] ;
static int adr4ibuf = 0 ;

void pushReal4(float x) {
  addBufferTraffic(4) ;
  adr4buf[adr4ibuf] = x ;
  if (adr4ibuf>=R4BUFSIZE-1) {
    pushNArray((char *)adr4buf, R4BUFSIZE*4, 1) ;
    addBufferTraffic(-R4BUFSIZE*4) ;
    adr4ibuf = 0 ;
  } else
    ++adr4ibuf ;
}

void popReal4(float *x) {
  if (adr4ibuf<=0) {
    popNArray((char *)adr4buf, R4BUFSIZE*4, 1) ;
    adr4ibuf = R4BUFSIZE-1 ;
  } else
    --adr4ibuf ;
  *x = adr4buf[adr4ibuf] ;
}

/************************** real*8 ************************/
// The buffer array for r8. Suggested size 512
#define R8BUFSIZE 512
static double adr8buf[R8BUFSIZE] ;
static int adr8ibuf = 0 ;

void pushReal8(double x) {
  addBufferTraffic(8) ;
  adr8buf[adr8ibuf] = x ;
  if (adr8ibuf>=R8BUFSIZE-1) {
    pushNArray((char *)adr8buf, R8BUFSIZE*8, 1) ;
    addBufferTraffic(-R8BUFSIZE*8) ;
    adr8ibuf = 0 ;
  } else
    ++adr8ibuf ;
}

void popReal8(double *x) {
  if (adr8ibuf<=0) {
    popNArray((char *)adr8buf, R8BUFSIZE*8, 1) ;
    adr8ibuf = R8BUFSIZE-1 ;
  } else
    --adr8ibuf ;
  *x = adr8buf[adr8ibuf] ;
}

/************************** complex*8 ************************/
// The buffer array for C8. Suggested size 512
#define C8BUFSIZE 512
static ccmplx adc8buf[C8BUFSIZE] ;
static int adc8ibuf = 0 ;

void pushComplex8(ccmplx x) {
  addBufferTraffic(8) ;
  adc8buf[adc8ibuf] = x ;
  if (adc8ibuf>=C8BUFSIZE-1) {
    pushNArray((char *)adc8buf, C8BUFSIZE*8, 1) ;
    addBufferTraffic(-C8BUFSIZE*8) ;
    adc8ibuf = 0 ;
  } else
  ++adc8ibuf ;
}

void popComplex8(ccmplx *x) {
  if (adc8ibuf<=0) {
    popNArray((char *)adc8buf, C8BUFSIZE*8, 1) ;
    adc8ibuf = C8BUFSIZE-1 ;
  } else
    --adc8ibuf ;
  *x = adc8buf[adc8ibuf] ;
}

/************************** complex*16 ************************/
// The buffer array for C16. Suggested size 512
#define C16BUFSIZE 512
static cdcmplx adc16buf[C16BUFSIZE] ;
static int adc16ibuf = 0 ;

void pushComplex16(cdcmplx x) {
  addBufferTraffic(16) ;
  adc16buf[adc16ibuf] = x ;
  if (adc16ibuf>=C16BUFSIZE-1) {
    pushNArray((char *)adc16buf, C16BUFSIZE*16, 1) ;
    addBufferTraffic(-C16BUFSIZE*16) ;
    adc16ibuf = 0 ;
  } else
    ++adc16ibuf ;
}

void popComplex16(cdcmplx *x) {
  if (adc16ibuf<=0) {
    popNArray((char *)adc16buf, C16BUFSIZE*16, 1) ;
    adc16ibuf = C16BUFSIZE-1 ;
  } else
    --adc16ibuf ;
  *x = adc16buf[adc16ibuf] ;
}

/************************** character ************************/
// The buffer array for characters. Suggested size 512
#define CHARBUFSIZE 512
static char ads1buf[CHARBUFSIZE] ;
static int ads1ibuf = 0 ;

void pushCharacter(char x) {
  addBufferTraffic(1) ;
  ads1buf[ads1ibuf] = x ;
  if (ads1ibuf>=CHARBUFSIZE-1) {
    pushNArray((char *)ads1buf, CHARBUFSIZE, 1) ;
    addBufferTraffic(-CHARBUFSIZE) ;
    ads1ibuf = 0 ;
  } else
    ++ads1ibuf ;
}

void popCharacter(char *x) {
  if (ads1ibuf<=0) {
    popNArray((char *)ads1buf, CHARBUFSIZE, 1) ;
    ads1ibuf = CHARBUFSIZE-1 ;
  } else
    --ads1ibuf ;
  *x = ads1buf[ads1ibuf] ;
}

/******************* bit (hidden primitives) ***************/
static unsigned int adbitbuf = 0 ;
static int adbitibuf = 0 ;

void pushBit(int x) {
  adbitbuf<<=1 ;
  if (x) ++adbitbuf ;
  if (adbitibuf>=31) {
    pushNArray((char *)&adbitbuf, 4, 1) ;
    adbitbuf = 0 ;
    adbitibuf = 0 ;
  } else
    ++adbitibuf ;
}

int popBit() {
  if (adbitibuf<=0) {
    popNArray((char *)&adbitbuf, 4, 1) ;
    adbitibuf = 31 ;
  } else
    --adbitibuf ;
  int result = adbitbuf%2 ;
  adbitbuf>>=1 ;
  return result ;
}

/*************************** boolean *************************/

void pushBoolean(int x) {
  pushBit(x) ;
}

void popBoolean(int *x) {
  *x = popBit() ;
}

/************************* control ***********************/

void pushControl1b(int cc) {
  pushBit(cc) ;
}

void popControl1b(int *cc) {
  *cc = popBit() ;
}

void pushControl2b(int cc) {
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc) ;
}

void popControl2b(int *cc) {
  *cc = (popBit()?2:0) ;
  if (popBit()) (*cc)++ ;
}

void pushControl3b(int cc) {
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc) ;
}

void popControl3b(int *cc) {
  *cc = (popBit()?2:0) ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
}

void pushControl4b(int cc) {
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc) ;
}

void popControl4b(int *cc) {
  *cc = (popBit()?2:0) ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
}

void pushControl5b(int cc) {
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc) ;
}

void popControl5b(int *cc) {
  *cc = (popBit()?2:0) ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
}

void pushControl6b(int cc) {
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc) ;
}

void popControl6b(int *cc) {
  *cc = (popBit()?2:0) ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
}

void pushControl7b(int cc) {
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc) ;
}

void popControl7b(int *cc) {
  *cc = (popBit()?2:0) ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
}

void pushControl8b(int cc) {
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc%2) ;
  cc>>=1 ;
  pushBit(cc) ;
}

void popControl8b(int *cc) {
  *cc = (popBit()?2:0) ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
  (*cc) <<= 1 ;
  if (popBit()) (*cc)++ ;
}

/************************* pointer ************************/
// The buffer array for pointers. Suggested size PTRBUFSIZE 512
// Depending on the system, these use 4 or 8 bytes,
// but they are all 4 or all 8, never a mixture of both.
#define PTRBUFSIZE 512
static void * adptrbuf[PTRBUFSIZE] ;
static int adptribuf = 0 ;

void pushPointer4(void *x) {
  addBufferTraffic(4) ;
  adptrbuf[adptribuf] = x ;
  if (adptribuf>=PTRBUFSIZE-1) {
    pushNArray((char *)adptrbuf, PTRBUFSIZE*4, 1) ;
    addBufferTraffic(-PTRBUFSIZE*4) ;
    adptribuf = 0 ;
  } else
    ++adptribuf ;
}

void popPointer4(void **x) {
  if (adptribuf<=0) {
    popNArray((char *)adptrbuf, PTRBUFSIZE*4, 1) ;
    adptribuf = PTRBUFSIZE-1 ;
  } else
    --adptribuf ;
  *x = adptrbuf[adptribuf] ;
}

void pushPointer8(void *x) {
  addBufferTraffic(8) ;
  adptrbuf[adptribuf] = x ;
  if (adptribuf>=PTRBUFSIZE-1) {
    pushNArray((char *)adptrbuf, PTRBUFSIZE*8, 1) ;
    addBufferTraffic(-PTRBUFSIZE*8) ;
    adptribuf = 0 ;
  } else
    ++adptribuf ;
}

void popPointer8(void **x) {
  if (adptribuf<=0) {
    popNArray((char *)adptrbuf, PTRBUFSIZE*8, 1) ;
    adptribuf = PTRBUFSIZE-1 ;
  } else
    --adptribuf ;
  *x = adptrbuf[adptribuf] ;
}

/**********************************************************
 *        HOW TO CREATE PUSH* POP* SUBROUTINES
 *             YET FOR OTHER DATA TYPES
 * Duplicate and uncomment the commented code below.
 * In the duplicated and uncommented code, replace:
 *   ctct -> C type name (e.g. float double, int...)
 *   tttt -> BASIC TAPENADE TYPE NAME
 *     (in character, boolean, integer, real, complex, pointer,...)
 *   z7   -> LETTERSIZE FOR TYPE
 *     (LETTER in s, b, i, r, c, p, ...) (SIZE is type size in bytes)
 *   7    -> TYPE SIZE IN BYTES
 **********************************************************/

/************************** tttt*7 ************************/
/*
// The buffer array for Z7. Suggested size 512
#define Z7BUFSIZE 512
static ctct adz7buf[Z7BUFSIZE] ;
static ctct adz7ibuf = 0 ;

void pushTttt7(ctct x) {
  addBufferTraffic(7) ;
  adz7buf[adz7ibuf] = x ;
  if (adz7ibuf>=Z7BUFSIZE-1) {
    pushNArray((char *)adz7buf, Z7BUFSIZE*7, 1) ;
    addBufferTraffic(-Z7BUFSIZE*7) ;
    adz7ibuf = 0 ;
  } else
    ++adz7ibuf ;
}

void popTttt7(ctct *x) {
  if (adz7ibuf <= 0) {
    popNArray((char *)adz7buf, Z7BUFSIZE*7, 1) ;
    adz7ibuf = Z7BUFSIZE-1 ;
  } else
    --adz7ibuf ;
  *x = adz7buf[adz7ibuf] ;
}

void pushTttt7Array(ctct *x, int n) {
  pushNArray((char *)x,(unsigned int)(n*7), 1) ;
}

void popTttt7Array(ctct *x, int n) {
  popNArray((char *)x,(unsigned int)(n*7), 1) ;
}
*/

/*************** REPEATED ACCESS MECHANISM *********************/

typedef struct _BufferRepeatCell {
  int indexi4 ;
  int indexi8 ;
  int indexr4 ;
  int indexr8 ;
  int indexc8 ;
  int indexc16 ;
  int indexs1 ;
  int indexbit ;
  int indexptr ;
  struct _BufferRepeatCell *previous ;
} BufferRepeatCell ;

BufferRepeatCell *bufferRepeatTop = NULL ;

void adStack_startRepeat() {
  // Create (push) a new "buffers" repeat level:
  BufferRepeatCell *newRepeatCell = (BufferRepeatCell *)malloc(sizeof(BufferRepeatCell)) ;
  newRepeatCell->previous = bufferRepeatTop ;
  // Also create (push) a new repeat level for the main stack:
  startStackRepeat1() ;
  // Push all local buffers on the main stack
  // 3rd arg is 0 to deactivate the check for stack read-only zone:
  pushNArray((char *)adi4buf, adi4ibuf*4, 0) ;
  pushNArray((char *)adi8buf, adi8ibuf*8, 0) ;
  pushNArray((char *)adr4buf, adr4ibuf*4, 0) ;
  pushNArray((char *)adr8buf, adr8ibuf*8, 0) ;
  pushNArray((char *)adc8buf, adc8ibuf*sizeof(ccmplx), 0) ;
  pushNArray((char *)adc16buf, adc16ibuf*sizeof(cdcmplx), 0) ;
  pushNArray((char *)ads1buf, ads1ibuf, 0) ;
  pushNArray((char *)&adbitbuf, 4, 0) ;
  pushNArray((char *)adptrbuf, adptribuf*sizeof(void *), 0) ;
  newRepeatCell->indexi4 = adi4ibuf ;
  newRepeatCell->indexi8 = adi8ibuf ;
  newRepeatCell->indexr4 = adr4ibuf ;
  newRepeatCell->indexr8 = adr8ibuf ;
  newRepeatCell->indexc8 = adc8ibuf ;
  newRepeatCell->indexc16 = adc16ibuf ;
  newRepeatCell->indexs1 = ads1ibuf ;
  newRepeatCell->indexbit = adbitibuf ;
  newRepeatCell->indexptr = adptribuf ;
  // Store current location as repeat location of new repeat level.
  // Note that this repeat location protects below as read-only.
  // Make the new repeat level the current repeat level  for the main stack:
  startStackRepeat2() ;
  // Make this new repeat level the current repeat level:
  bufferRepeatTop = newRepeatCell ;
}

// Note: adStack_resetrepeat() forces exit from any internal
//  checkpointed sequence, i.e. all nested push'es are forced popped.
void adStack_resetRepeat() {
  // First stage of reset repeat for the main stack:
  resetStackRepeat1() ;
  // Restore all local buffers:
  adi4ibuf  = bufferRepeatTop->indexi4 ;
  adi8ibuf  = bufferRepeatTop->indexi8 ;
  adr4ibuf  = bufferRepeatTop->indexr4 ;
  adr8ibuf  = bufferRepeatTop->indexr8 ;
  adc8ibuf  = bufferRepeatTop->indexc8 ;
  adc16ibuf = bufferRepeatTop->indexc16 ;
  ads1ibuf  = bufferRepeatTop->indexs1 ;
  adbitibuf = bufferRepeatTop->indexbit ;
  adptribuf = bufferRepeatTop->indexptr ;
  // 3rd arg is 0 to deactivate the check for stack read-only zone:
  popNArray((char *)adptrbuf, adptribuf*sizeof(void *), 0) ;
  popNArray((char *)&adbitbuf, 4, 0) ;
  popNArray((char *)ads1buf,  ads1ibuf, 0) ;
  popNArray((char *)adc16buf, adc16ibuf*sizeof(cdcmplx), 0) ;
  popNArray((char *)adc8buf,  adc8ibuf*sizeof(ccmplx), 0) ;
  popNArray((char *)adr8buf,  adr8ibuf*8, 0) ;
  popNArray((char *)adr4buf,  adr4ibuf*4, 0) ;
  popNArray((char *)adi8buf,  adi8ibuf*8, 0) ;
  popNArray((char *)adi4buf,  adi4ibuf*4, 0) ;
  // Second stage of reset repeat for the main stack:
  resetStackRepeat2() ;
}

// Note: adStack_endrepeat() forces exit from any internal
//  checkpointed sequence, i.e. all nested push'es are forced popped.
void adStack_endRepeat() {
  // Remove (pop) top repeat level for the main stack:
  endStackRepeat() ;
  // Remove (pop) top "buffer" repeat level:
  BufferRepeatCell *oldRepeatCell = bufferRepeatTop ;
  bufferRepeatTop = bufferRepeatTop->previous ;
  free(oldRepeatCell) ;
}

void showBufferRepeatsRec(BufferRepeatCell *inRepeatStack, int type) {
  if (inRepeatStack->previous) {showBufferRepeatsRec(inRepeatStack->previous, type) ; printf(" ; ") ;}
  switch (type) {
  case 1:
    printf("%1i", inRepeatStack->indexi4) ;
    break ;
  case 2:
    printf("%1i", inRepeatStack->indexi8) ;
    break ;
  case 3:
    printf("%1i", inRepeatStack->indexr4) ;
    break ;
  case 4:
    printf("%1i", inRepeatStack->indexr8) ;
    break ;
  case 5:
    printf("%1i", inRepeatStack->indexc8) ;
    break ;
  case 6:
    printf("%1i", inRepeatStack->indexc16) ;
    break ;
  case 7:
    printf("%1i", inRepeatStack->indexs1) ;
    break ;
  case 8:
    printf("%1i", inRepeatStack->indexbit) ;
    break ;
  case 9:
    printf("%1i", inRepeatStack->indexptr) ;
    break ;
  }
}

void showBufferRepeats(BufferRepeatCell *inRepeatStack, int type) {
  printf("        REPEATS:") ;
  if (inRepeatStack)
    showBufferRepeatsRec(inRepeatStack, type) ;
  else
    printf(" none!") ;
}

void showStackAndBuffers(char *locationName) {
  int i ;
  printf("%6s: ", locationName) ;
  showStack() ;
  printf("        I4:") ;
  for (i=0 ; i<I4BUFSIZE ; ++i) {
    if (i==adi4ibuf) printf(" | ") ;
    printf(" %11i",adi4buf[i]) ;
  }
  showBufferRepeats(bufferRepeatTop, 1) ;
  printf("\n") ;
  printf("        I8:") ;
  for (i=0 ; i<I8BUFSIZE ; ++i) {
    if (i==adi8ibuf) printf(" | ") ;
    printf(" %11i",adi8buf[i]) ;
  }
  showBufferRepeats(bufferRepeatTop, 2) ;
  printf("\n") ;
  printf("        R4:") ;
  for (i=0 ; i<R4BUFSIZE ; ++i) {
    if (i==adr4ibuf) printf(" | ") ;
    printf(" %f",adr4buf[i]) ;
  }
  showBufferRepeats(bufferRepeatTop, 3) ;
  printf("\n") ;
  printf("        R8:") ;
  for (i=0 ; i<R8BUFSIZE ; ++i) {
    if (i==adr8ibuf) printf(" | ") ;
    printf(" %f",adr8buf[i]) ;
  }
  showBufferRepeats(bufferRepeatTop, 4) ;
  printf("\n") ;
  printf("        C8:") ;
  for (i=0 ; i<C8BUFSIZE ; ++i) {
    if (i==adc8ibuf) printf(" | ") ;
    printf(" %f",adc8buf[i]) ;
  }
  showBufferRepeats(bufferRepeatTop, 5) ;
  printf("\n") ;
  printf("        C16:") ;
  for (i=0 ; i<C16BUFSIZE ; ++i) {
    if (i==adc16ibuf) printf(" | ") ;
    printf(" %f",adc16buf[i]) ;
  }
  showBufferRepeats(bufferRepeatTop, 6) ;
  printf("\n") ;
  printf("        STR:") ;
  for (i=0 ; i<CHARBUFSIZE ; ++i) {
    if (i==ads1ibuf) printf(" | ") ;
    printf(" %c",ads1buf[i]) ;
  }
  showBufferRepeats(bufferRepeatTop, 7) ;
  printf("\n") ;
  printf("        BITS:%1i in %08x", adbitibuf, adbitbuf) ;
  showBufferRepeats(bufferRepeatTop, 8) ;
  printf("\n") ;
  printf("        PTR:") ;
  for (i=0 ; i<PTRBUFSIZE ; ++i) {
    if (i==adptribuf) printf(" | ") ;
    printf(" %x",adptrbuf[i]) ;
  }
  showBufferRepeats(bufferRepeatTop, 9) ;
  printf("\n") ;
}

void showStackAndBuffersSize(int pos) {
  showStackSize(adi4ibuf,adi8ibuf,adr4ibuf,adr8ibuf,adc8ibuf,adc16ibuf,ads1ibuf,adbitibuf,adptribuf,pos) ;
}




'''
#ifndef OCV_UTILS_H
#define OCV_UTILS_H

#include <sys/time.h>
#include <unistd.h>

#define TIME_COUNT( msg, code_fragment ) \
{ \
	timeval ti, tf; \
	gettimeofday(&ti, NULL); \
\
	code_fragment \
\	
	gettimeofday(&tf, NULL); \
	printf(msg"\t%ld", tf.tv_usec - ti.tv_usec); \
}


#define SLEEP_1_MS usleep(1000);


#endif

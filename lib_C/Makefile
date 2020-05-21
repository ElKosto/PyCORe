CC = gcc
CFLAGS = -fPIC -Wall -Wextra -O3 -g -lfftw3 -lm
LDFLAGS = -shared
TARGET_LIB = lib_lle_core.so
SRCS = lle_core.c
OBJS = $(SRCS:.c=.o)

.PHONY: all
all: ${TARGET_LIB}

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -o $@ $^

$(SRCS:.c=.d):%.d:%.c
	$(CC) $(CFLAGS) -MM $< >$@

include $(SRCS:.c=.d)

.PHONY: clean
clean:
	-${RM} ${TARGET_LIB} ${OBJS} $(SRCS:.c=.d)


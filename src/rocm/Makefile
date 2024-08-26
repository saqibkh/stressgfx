# Makefile for compiling main.cpp and add.hip

CC = hipcc
CFLAGS = -O2
TARGET = run 

all: $(TARGET)

$(TARGET): main.o add.o subtract.o
	$(CC) $(CFLAGS) -o $(TARGET) main.o add.o subtract.o

main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp

add.o: add.cpp
	$(CC) $(CFLAGS) -c add.cpp

subtract.o: subtract.cpp
	$(CC) $(CFLAGS) -c subtract.cpp

clean:
	rm -f *.o $(TARGET)

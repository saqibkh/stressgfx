# Makefile for compiling main.cpp and add.hip

CC = hipcc
CFLAGS = -O2 -Wno-implicit-const-int-float-conversion -Wno-unused-result 
TARGET = run 

all: $(TARGET)

$(TARGET): main.o add.o subtract.o multiply.o divide.o
	$(CC) $(CFLAGS) -o $(TARGET) main.o add.o subtract.o multiply.o divide.o

main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp

add.o: add.cpp
	$(CC) $(CFLAGS) -c add.cpp

subtract.o: subtract.cpp
	$(CC) $(CFLAGS) -c subtract.cpp

multiply.o: multiply.cpp
	$(CC) $(CFLAGS) -c multiply.cpp
	
divide.o: divide.cpp
	$(CC) $(CFLAGS) -c divide.cpp

clean:
	rm -f *.o $(TARGET)

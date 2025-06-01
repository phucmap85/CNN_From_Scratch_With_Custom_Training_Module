# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -O2 -g

# Target executable
TARGET = train

# For deleting the target
TARGET_DEL = train.exe

# Source files
SRCS = train.cpp include\\lodepng\\lodepng.cpp include\\nn\\nn.cpp include\\dataloader\\dataloader.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default rule to build and run the executable
all: $(TARGET) run

# Rule to link object files into the target executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule to compile .cpp files into .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to run the executable
run: $(TARGET)
	$(TARGET)

# Clean rule to remove generated files
clean:
	del $(TARGET_DEL) $(OBJS)
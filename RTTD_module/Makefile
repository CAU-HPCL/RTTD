ifndef CONDA_PREFIX
$(error CONDA_PREFIX is not set. Please activate your conda environment first!)
endif

NVCC   = nvcc
PYTHON = $(CONDA_PREFIX)/bin/python

PYTHON_LDFLAGS := $(shell $(PYTHON)-config --ldflags 2>/dev/null || python3-config --ldflags 2>/dev/null || echo "")
PYBIND11_INCLUDE := $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

CUDA_INCLUDE = -I$(CONDA_PREFIX)/include
CUDA_LIB     = -L$(CONDA_PREFIX)/lib -lcudart -lcublas -lcusolver -lcurand
RPATH_FLAGS  = -Xlinker -rpath,$(CONDA_PREFIX)/lib

ARCH_FLAGS = \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_90,code=compute_90

NVCC_FLAGS = -std=c++17 -O3 --compiler-options '-fPIC' $(ARCH_FLAGS) $(PYBIND11_INCLUDE) $(CUDA_INCLUDE)

TARGET = rttd_cpp_module$(EXT_SUFFIX)
SOURCE = rttd_cpp.cu

EXPERIMENTAL_DIR := $(shell $(PYTHON) -c "import os, cuquantum.cutensornet.experimental as e; print(os.path.dirname(e.__file__))")

.PHONY: all install clean print-dirs

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $< $(PYTHON_LDFLAGS) $(CUDA_LIB) $(RPATH_FLAGS)

install: $(TARGET)
	@echo "Installing into: $(EXPERIMENTAL_DIR)"
	@test -d "$(EXPERIMENTAL_DIR)" || (echo "ERROR: $(EXPERIMENTAL_DIR) not found" && exit 1)
	# copy the extension
	cp -f $(TARGET) "$(EXPERIMENTAL_DIR)/"
	# copy your python sources from *this* folder
	cp -f __init__.py rttd_network.py "$(EXPERIMENTAL_DIR)/"
	@echo "Done."

print-dirs:
	@echo "CONDA_PREFIX=$(CONDA_PREFIX)"
	@echo "PYTHON=$(PYTHON)"
	@echo "EXT_SUFFIX=$(EXT_SUFFIX)"
	@echo "EXPERIMENTAL_DIR=$(EXPERIMENTAL_DIR)"

clean:
	rm -f $(TARGET)
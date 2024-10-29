KOKKOS_PATH = ${HOME}/kokkos
KOKKOS_DEVICES = "OpenMP"
EXE_NAME = fet

SRC = finite_element_tetrahedral.cpp

default: build
	echo "Start Build"


ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
SUFFIX = cuda
KOKKOS_ARCH = "Turing75"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
else
CXX = g++
SUFFIX = host
KOKKOS_ARCH = "SNB"
endif

VARIANTS = float_left float_right double_left double_right
BASE_STEMS = $(addprefix $(EXE_NAME)_,$(VARIANTS))

CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

OBJ = $(addsuffix .o,$(BASE_STEMS))
EXE = $(addsuffix .$(SUFFIX),$(BASE_STEMS))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

# Separate rules for each executable
fet_float_left.$(SUFFIX): fet_float_left.o $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) fet_float_left.o $(KOKKOS_LIBS) $(LIB) -o $@

fet_float_right.$(SUFFIX): fet_float_right.o $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) fet_float_right.o $(KOKKOS_LIBS) $(LIB) -o $@

fet_double_left.$(SUFFIX): fet_double_left.o $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) fet_double_left.o $(KOKKOS_LIBS) $(LIB) -o $@

fet_double_right.$(SUFFIX): fet_double_right.o $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) fet_double_right.o $(KOKKOS_LIBS) $(LIB) -o $@


clean: kokkos-clean
	rm -f *.o *.cuda *.host *.tmp

# Compilation rules

fet_float_left.o: $(SRC) $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -DNumberType=float -DMemLayout=Kokkos::LayoutLeft -c $< -o $@

fet_float_right.o: $(SRC) $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -DNumberType=float -DMemLayout=Kokkos::LayoutRight -c $< -o $@

fet_double_left.o: $(SRC) $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -DNumberType=double -DMemLayout=Kokkos::LayoutLeft -c $< -o $@

fet_double_right.o: $(SRC) $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -DNumberType=double -DMemLayout=Kokkos::LayoutRight -c $< -o $@

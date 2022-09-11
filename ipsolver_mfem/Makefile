# Use the MFEM build directory
MFEM_DIR ?= /home/tucker/software/glvis_mfem/mfem/
MFEM_BUILD_DIR ?= /home/tucker/software/glvis_mfem/mfem/
SRC = ./

CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# Remove built-in rule
#%: %.cpp

diffusionDriver: diffusionDriver.o problems.o helper.o IPsolver.o $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) diffusionDriver.o problems.o helper.o IPsolver.o -o $@ $(MFEM_LIBS)

problems.o: problems.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

IPsolver.o: IPsolver.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

diffusionDriver.o: diffusionDriver.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

helper.o: helper.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 




# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

.PHONY: clean
clean:
	rm -f *.o scratch scratch2 scratch3 FDcheck diffusionDriver



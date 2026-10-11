# Build all tests (unit + integration).
# Works on Windows and Unix systems.
# Usage (from this directory):
#   make              # build everything into bin/
#   make unit         # build unit tests
#   make integration  # build integration tests
#   make test         # build and run unit tests
#   make clean

# Make's default CXX is g++; prefer clang++ unless the user overrides.
ifeq ($(origin CXX),default)
    CXX := clang++
endif
CXXFLAGS ?= -std=c++17 -O2 -march=native -Wall

ROOT := ..
BIN  := bin

ifeq ($(OS),Windows_NT)
    EXEEXT  := .exe
    MKDIR_P := if not exist $(BIN) mkdir $(BIN)
    RM_BIN  := if exist $(BIN) del /Q $(BIN)\*$(EXEEXT)
else
    EXEEXT  :=
    MKDIR_P := mkdir -p $(BIN)
    RM_BIN  := rm -f $(BIN)/*$(EXEEXT)
endif

# Recursive wildcard: $(call rwildcard,dir,*.cpp)
rwildcard = $(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

# All .cpp under test/, except integration/ (and bin/)
ALL_SRCS         := $(call rwildcard,.,*.cpp)
UNIT_SRCS        := $(filter-out ./integration/% ./bin/%,$(ALL_SRCS))
INTEGRATION_SRCS := $(filter ./integration/%,$(ALL_SRCS))
SRCS             := $(UNIT_SRCS) $(INTEGRATION_SRCS)

# Headers (forward-slash paths). Compilers' -MMD .d files use backslashes on
# Windows, which Make treats as escapes, so we track headers explicitly.
HDRS := \
    $(call rwildcard,$(ROOT)/neural,*.hpp) \
    $(call rwildcard,$(ROOT)/lin_alg,*.hpp) \
    test.hpp

UNIT_NAMES        := $(basename $(notdir $(UNIT_SRCS)))
INTEGRATION_NAMES := $(basename $(notdir $(INTEGRATION_SRCS)))
UNIT_EXES         := $(addprefix $(BIN)/,$(addsuffix $(EXEEXT),$(UNIT_NAMES)))
INTEGRATION_EXES  := $(addprefix $(BIN)/,$(addsuffix $(EXEEXT),$(INTEGRATION_NAMES)))
RUN_UNIT          := $(addprefix run-,$(UNIT_NAMES))

.PHONY: all unit integration test clean $(RUN_UNIT)

all: unit integration

unit: $(UNIT_EXES)

integration: $(INTEGRATION_EXES)

# Run from repo root so relative paths like data/ and test/bin/ resolve.
test: $(RUN_UNIT)

ifeq ($(OS),Windows_NT)
define RUN_RULE
run-$(1): $(BIN)/$(1)$(EXEEXT)
	cd $$(ROOT) && test\bin\$(1)$$(EXEEXT)
endef
else
define RUN_RULE
run-$(1): $(BIN)/$(1)$(EXEEXT)
	cd $$(ROOT) && ./test/bin/$(1)$$(EXEEXT)
endef
endif
$(foreach name,$(UNIT_NAMES),$(eval $(call RUN_RULE,$(name))))

$(BIN):
	@$(MKDIR_P)

define TEST_RULE
$(BIN)/$(basename $(notdir $(1)))$(EXEEXT): $(1) $(HDRS) | $(BIN)
	$$(CXX) $$(CXXFLAGS) -o $$@ $$<
endef
$(foreach src,$(SRCS),$(eval $(call TEST_RULE,$(src))))

clean:
	@$(RM_BIN)

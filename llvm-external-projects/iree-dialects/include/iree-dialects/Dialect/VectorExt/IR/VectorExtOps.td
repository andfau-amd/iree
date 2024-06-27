// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VECTOREXT_OPS
#define IREE_DIALECT_VECTOREXT_OPS

include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.td"
include "iree-dialects/Dialect/VectorExt/IR/VectorExtBase.td"
include "iree-dialects/Dialect/VectorExt/IR/VectorExtInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class IREEVectorExt_PureOp<string mnemonic, list<Trait> traits = []> :
    Op<IREEVectorExt_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Layout ops.
//===----------------------------------------------------------------------===//

def IREEVectorExt_LayoutConflictResolutionOp : IREEVectorExt_PureOp<"layout_conflict_resolution"> {
  let summary = "Layout Conflict Resolution operator";
  let description = [{
    The layout conflict resolution operator takes a vector and a
    desired layout and transforms the vector to one with the
    desired layout.
  }];
  let arguments = (ins
    AnyVector:$input,
    VectorLayoutInterface:$sourceLayout,
    VectorLayoutInterface:$desiredLayout
  );
  let results = (outs
    AnyVector:$output
  );
  let extraClassDeclaration = [{}];
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

def IREEVectorExt_ToSIMDOp : IREEVectorExt_PureOp<"to_simd",
    [SameOperandsAndResultElementType, Pure]> {
  let summary = "SIMT to SIMD conversion operation";
  let description = [{
    This operation is a temporary operation useful for source/target
    materializations when doing type conversions between distributed and not
    distributed vectors.
  }];
  let arguments = (ins
    AnyVector:$input
  );
  let results = (outs
    AnyVector:$output
  );
  let extraClassDeclaration = [{}];
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasFolder = 1;
}

def IREEVectorExt_ToSIMTOp : IREEVectorExt_PureOp<"to_simt",
    [SameOperandsAndResultElementType, Pure]> {
  let summary = "SIMD to SIMT conversion operation";
  let description = [{
    This operation is a temporary operation useful for source/target
    materializations when doing type conversions between distributed and not
    distributed vectors.
  }];
  let arguments = (ins
    AnyVector:$input
  );
  let results = (outs
    AnyVector:$output
  );
  let extraClassDeclaration = [{}];
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasFolder = 1;
}

def IREEVectorExt_MultiDimReductionWithIndexOp :
  IREEVectorExt_PureOp<"multi_reduction_index", [Pure,
    AllTypesMatch<["dest_value", "acc_value"]>,
    AllTypesMatch<["dest_index", "acc_index"]>,
    PredOpTrait<"source operand and result have same element type",
                TCresVTEtIsSameAsOpBase<0, 0>>,
//    DeclareOpInterfaceMethods<InferTypeOpInterface>,
//    DeclareOpInterfaceMethods<MaskableOpInterface>,
//    DeclareOpInterfaceMethods<VectorUnrollOpInterface,
//                              ["getShapeForUnroll"]>
]>,
    Arguments<(ins ReductionWithIndexKindAttr:$kind,
                   AnyVector:$source,
                   AnyType:$acc_value,
                   AnyTypeOf<[AnyInteger, Index]>:$acc_index,
                   I64ArrayAttr:$reduction_dims)>,
    Results<(outs AnyType:$dest_value,
                  AnyTypeOf<[AnyInteger, Index]>:$dest_index)> {
  let summary = "Multi-dimensional reduction operation returning index with value";
  let description = [{
    `vector.multi_reduction`, but the index of the original element is returned
    with the value. This allows implementing the argmax and argmin operations.
    Supported operations are `minsi`/`minui`/`maxsi`/`maxui` for integers, and
    `minnumf`/`maxnumf`/`minimumf`/`maximumf` for floats.
    Takes initial accumulator operands for the value and index.

    TODO: Example
  }];
  let builders = [
    OpBuilder<(ins "Value":$source, "Value":$acc_value, "Value":$acc_index,
                   "ArrayRef<bool>":$reductionMask, "ReductionWithIndexKind":$kind)>
  ];
  let extraClassDeclaration = [{
    VectorType getSourceVectorType() {
      return ::llvm::cast<VectorType>(getSource().getType());
    }
    Type getDestValueType() {
      return getDestValue().getType();
    }
    Type getDestIndexType() {
      return getDestIndex().getType();
    }

    bool isReducedDim(int64_t d) {
      assert(d >= 0 && d < static_cast<int64_t>(getReductionMask().size()) &&
        "d overflows the number of dims");
      return getReductionMask()[d];
    }

    SmallVector<bool> getReductionMask() {
      SmallVector<bool> res(getSourceVectorType().getRank(), false);
      for (auto ia : getReductionDims().getAsRange<IntegerAttr>())
        res[ia.getInt()] = true;
      return res;
    }
    static SmallVector<bool> getReductionMask(
        ArrayRef<int64_t> reductionDims, unsigned sourceRank) {
      SmallVector<bool> res(sourceRank, false);
      for (auto idx : reductionDims)
        res[idx] = true;
      return res;
    }
  }];
  let assemblyFormat =
    "$kind `,` $source `,` $acc_value `,` $acc_index attr-dict $reduction_dims `:` type($source) `to` type($dest_value) `,` type($dest_index)";
  //let hasFolder = 1;
  //let hasCanonicalizer = 1;
  //let hasVerifier = 1;
}


#endif  // IREE_DIALECT_VECTOREXT_OPS

░░
Ў(═(
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	АР
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetypeИ
E
AssignSubVariableOp
resource
value"dtype"
dtypetypeИ
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ъ
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
B
Equal
x"T
y"T
z
"
Ttype:
2	
Р
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
Ш
FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%╖╤8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
╘
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
л
ResourceApplyKerasMomentum
var	
accum
lr"T	
grad"T
momentum"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ
9
VarIsInitializedOp
resource
is_initialized
И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.14.02v1.14.0-0-g87989f6959ш▒

\
keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 
~
input_1Placeholder*&
shape:         АА*
dtype0*1
_output_shapes
:         АА
й
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"
   
          * 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
:
У
,conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *,Ч)╜* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
У
,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *,Ч)=* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
Ё
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape* 
_class
loc:@conv2d/kernel*
seed2 *
dtype0*&
_output_shapes
:

 *

seed *
T0
╥
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
ь
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*&
_output_shapes
:

 *
T0* 
_class
loc:@conv2d/kernel
▐
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*&
_output_shapes
:

 *
T0* 
_class
loc:@conv2d/kernel
▒
conv2d/kernelVarHandleOp*
shared_nameconv2d/kernel* 
_class
loc:@conv2d/kernel*
	container *
shape:

 *
dtype0*
_output_shapes
: 
k
.conv2d/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/kernel*
_output_shapes
: 
Р
conv2d/kernel/AssignAssignVariableOpconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
dtype0* 
_class
loc:@conv2d/kernel
Щ
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*&
_output_shapes
:

 
К
conv2d/bias/Initializer/zerosConst*
valueB *    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
Я
conv2d/biasVarHandleOp*
_output_shapes
: *
shared_nameconv2d/bias*
_class
loc:@conv2d/bias*
	container *
shape: *
dtype0
g
,conv2d/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/bias*
_output_shapes
: 

conv2d/bias/AssignAssignVariableOpconv2d/biasconv2d/bias/Initializer/zeros*
_class
loc:@conv2d/bias*
dtype0
З
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
e
conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
r
conv2d/Conv2D/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
:

 
В
conv2d/Conv2DConv2Dinput_1conv2d/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*1
_output_shapes
:         ўў 
e
conv2d/BiasAdd/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
: 
Ъ
conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*1
_output_shapes
:         ўў 
c
activation/ReluReluconv2d/BiasAdd*
T0*1
_output_shapes
:         ўў 
е
*batch_normalization/gamma/Initializer/onesConst*
_output_shapes
: *
valueB *  А?*,
_class"
 loc:@batch_normalization/gamma*
dtype0
╔
batch_normalization/gammaVarHandleOp*,
_class"
 loc:@batch_normalization/gamma*
	container *
shape: *
dtype0*
_output_shapes
: **
shared_namebatch_normalization/gamma
Г
:batch_normalization/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/gamma*
_output_shapes
: 
╢
 batch_normalization/gamma/AssignAssignVariableOpbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones*,
_class"
 loc:@batch_normalization/gamma*
dtype0
▒
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
dtype0*
_output_shapes
: *,
_class"
 loc:@batch_normalization/gamma
д
*batch_normalization/beta/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *+
_class!
loc:@batch_normalization/beta
╞
batch_normalization/betaVarHandleOp*)
shared_namebatch_normalization/beta*+
_class!
loc:@batch_normalization/beta*
	container *
shape: *
dtype0*
_output_shapes
: 
Б
9batch_normalization/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/beta*
_output_shapes
: 
│
batch_normalization/beta/AssignAssignVariableOpbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*+
_class!
loc:@batch_normalization/beta*
dtype0
о
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
: 
▓
1batch_normalization/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *2
_class(
&$loc:@batch_normalization/moving_mean
█
batch_normalization/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *0
shared_name!batch_normalization/moving_mean*2
_class(
&$loc:@batch_normalization/moving_mean*
	container *
shape: 
П
@batch_normalization/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/moving_mean*
_output_shapes
: 
╧
&batch_normalization/moving_mean/AssignAssignVariableOpbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros*
dtype0*2
_class(
&$loc:@batch_normalization/moving_mean
├
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0
╣
4batch_normalization/moving_variance/Initializer/onesConst*
valueB *  А?*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0*
_output_shapes
: 
ч
#batch_normalization/moving_varianceVarHandleOp*6
_class,
*(loc:@batch_normalization/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: *4
shared_name%#batch_normalization/moving_variance
Ч
Dbatch_normalization/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#batch_normalization/moving_variance*
_output_shapes
: 
▐
*batch_normalization/moving_variance/AssignAssignVariableOp#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0
╧
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
dtype0*
_output_shapes
: *6
_class,
*(loc:@batch_normalization/moving_variance
x
batch_normalization/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
q
!batch_normalization/cond/switch_tIdentity!batch_normalization/cond/Switch:1*
_output_shapes
: *
T0

o
!batch_normalization/cond/switch_fIdentitybatch_normalization/cond/Switch*
T0
*
_output_shapes
: 
c
 batch_normalization/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
Ф
'batch_normalization/cond/ReadVariableOpReadVariableOp0batch_normalization/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
: 
╞
.batch_normalization/cond/ReadVariableOp/SwitchSwitchbatch_normalization/gamma batch_normalization/cond/pred_id*
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
: : 
Ш
)batch_normalization/cond/ReadVariableOp_1ReadVariableOp2batch_normalization/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
: 
╞
0batch_normalization/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization/beta batch_normalization/cond/pred_id*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: : 
Е
batch_normalization/cond/ConstConst"^batch_normalization/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
З
 batch_normalization/cond/Const_1Const"^batch_normalization/cond/switch_t*
_output_shapes
: *
valueB *
dtype0
С
'batch_normalization/cond/FusedBatchNormFusedBatchNorm0batch_normalization/cond/FusedBatchNorm/Switch:1'batch_normalization/cond/ReadVariableOp)batch_normalization/cond/ReadVariableOp_1batch_normalization/cond/Const batch_normalization/cond/Const_1*
T0*
data_formatNHWC*I
_output_shapes7
5:         ўў : : : : *
is_training(*
epsilon%oГ:
ш
.batch_normalization/cond/FusedBatchNorm/SwitchSwitchactivation/Relu batch_normalization/cond/pred_id*
T0*"
_class
loc:@activation/Relu*N
_output_shapes<
::         ўў :         ўў 
Ц
)batch_normalization/cond/ReadVariableOp_2ReadVariableOp0batch_normalization/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
: 
╚
0batch_normalization/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization/gamma batch_normalization/cond/pred_id*
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
: : 
Ц
)batch_normalization/cond/ReadVariableOp_3ReadVariableOp0batch_normalization/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes
: 
╞
0batch_normalization/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization/beta batch_normalization/cond/pred_id*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: : 
┤
8batch_normalization/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOp?batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
: 
у
?batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitchbatch_normalization/moving_mean batch_normalization/cond/pred_id*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
: : 
╕
:batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpAbatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
: 
э
Abatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch#batch_normalization/moving_variance batch_normalization/cond/pred_id*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
: : 
╔
)batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm0batch_normalization/cond/FusedBatchNorm_1/Switch)batch_normalization/cond/ReadVariableOp_2)batch_normalization/cond/ReadVariableOp_38batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp:batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1*
data_formatNHWC*I
_output_shapes7
5:         ўў : : : : *
is_training( *
epsilon%oГ:*
T0
ъ
0batch_normalization/cond/FusedBatchNorm_1/SwitchSwitchactivation/Relu batch_normalization/cond/pred_id*"
_class
loc:@activation/Relu*N
_output_shapes<
::         ўў :         ўў *
T0
┬
batch_normalization/cond/MergeMerge)batch_normalization/cond/FusedBatchNorm_1'batch_normalization/cond/FusedBatchNorm*
T0*
N*3
_output_shapes!
:         ўў : 
▒
 batch_normalization/cond/Merge_1Merge+batch_normalization/cond/FusedBatchNorm_1:1)batch_normalization/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

: : 
▒
 batch_normalization/cond/Merge_2Merge+batch_normalization/cond/FusedBatchNorm_1:2)batch_normalization/cond/FusedBatchNorm:2*
N*
_output_shapes

: : *
T0
z
!batch_normalization/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization/cond_1/switch_tIdentity#batch_normalization/cond_1/Switch:1*
_output_shapes
: *
T0

s
#batch_normalization/cond_1/switch_fIdentity!batch_normalization/cond_1/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
Л
 batch_normalization/cond_1/ConstConst$^batch_normalization/cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *дp}?
Н
"batch_normalization/cond_1/Const_1Const$^batch_normalization/cond_1/switch_f*
_output_shapes
: *
valueB
 *  А?*
dtype0
Ы
 batch_normalization/cond_1/MergeMerge"batch_normalization/cond_1/Const_1 batch_normalization/cond_1/Const*
N*
_output_shapes
: : *
T0
в
)batch_normalization/AssignMovingAvg/sub/xConst*
_output_shapes
: *
valueB
 *  А?*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0
╨
'batch_normalization/AssignMovingAvg/subSub)batch_normalization/AssignMovingAvg/sub/x batch_normalization/cond_1/Merge*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
: 
О
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
dtype0*
_output_shapes
: 
▀
)batch_normalization/AssignMovingAvg/sub_1Sub2batch_normalization/AssignMovingAvg/ReadVariableOp batch_normalization/cond/Merge_1*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
: 
█
'batch_normalization/AssignMovingAvg/mulMul)batch_normalization/AssignMovingAvg/sub_1'batch_normalization/AssignMovingAvg/sub*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
: *
T0
┘
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpbatch_normalization/moving_mean'batch_normalization/AssignMovingAvg/mul*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0
■
4batch_normalization/AssignMovingAvg/ReadVariableOp_1ReadVariableOpbatch_normalization/moving_mean8^batch_normalization/AssignMovingAvg/AssignSubVariableOp*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0*
_output_shapes
: 
и
+batch_normalization/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
valueB
 *  А?*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0
╪
)batch_normalization/AssignMovingAvg_1/subSub+batch_normalization/AssignMovingAvg_1/sub/x batch_normalization/cond_1/Merge*
_output_shapes
: *
T0*6
_class,
*(loc:@batch_normalization/moving_variance
Ф
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
dtype0*
_output_shapes
: 
ч
+batch_normalization/AssignMovingAvg_1/sub_1Sub4batch_normalization/AssignMovingAvg_1/ReadVariableOp batch_normalization/cond/Merge_2*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
: 
х
)batch_normalization/AssignMovingAvg_1/mulMul+batch_normalization/AssignMovingAvg_1/sub_1)batch_normalization/AssignMovingAvg_1/sub*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
: 
х
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batch_normalization/moving_variance)batch_normalization/AssignMovingAvg_1/mul*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0
К
6batch_normalization/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp#batch_normalization/moving_variance:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0*
_output_shapes
: 
═
max_pooling2d/MaxPoolMaxPoolbatch_normalization/cond/Merge*/
_output_shapes
:         {{ *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
н
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"          @   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:
Ч
.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *%I╜*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
Ч
.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *%I=*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
Ў
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_1/kernel*
seed2 *
dtype0*&
_output_shapes
: @*

seed 
┌
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_1/kernel
Ї
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @
ц
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @
╖
conv2d_1/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_nameconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
	container *
shape: @
o
0conv2d_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
Ш
conv2d_1/kernel/AssignAssignVariableOpconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_1/kernel*
dtype0
Я
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
: @
О
conv2d_1/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
е
conv2d_1/biasVarHandleOp* 
_class
loc:@conv2d_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d_1/bias
k
.conv2d_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/bias*
_output_shapes
: 
З
conv2d_1/bias/AssignAssignVariableOpconv2d_1/biasconv2d_1/bias/Initializer/zeros* 
_class
loc:@conv2d_1/bias*
dtype0
Н
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
g
conv2d_1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
v
conv2d_1/Conv2D/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
: @
Т
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPoolconv2d_1/Conv2D/ReadVariableOp*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:         uu@*
	dilations
*
T0
i
conv2d_1/BiasAdd/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:@
Ю
conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:         uu@
e
activation_1/ReluReluconv2d_1/BiasAdd*/
_output_shapes
:         uu@*
T0
й
,batch_normalization_1/gamma/Initializer/onesConst*
valueB@*  А?*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
:@
╧
batch_normalization_1/gammaVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_1/gamma*.
_class$
" loc:@batch_normalization_1/gamma*
	container 
З
<batch_normalization_1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
╛
"batch_normalization_1/gamma/AssignAssignVariableOpbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0
╖
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
:@
и
,batch_normalization_1/beta/Initializer/zerosConst*
valueB@*    *-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
:@
╠
batch_normalization_1/betaVarHandleOp*+
shared_namebatch_normalization_1/beta*-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:@*
dtype0*
_output_shapes
: 
Е
;batch_normalization_1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
╗
!batch_normalization_1/beta/AssignAssignVariableOpbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_1/beta*
dtype0
┤
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
:@
╢
3batch_normalization_1/moving_mean/Initializer/zerosConst*
valueB@*    *4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:@
с
!batch_normalization_1/moving_meanVarHandleOp*
	container *
shape:@*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_1/moving_mean*4
_class*
(&loc:@batch_normalization_1/moving_mean
У
Bbatch_normalization_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
╫
(batch_normalization_1/moving_mean/AssignAssignVariableOp!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0
╔
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:@
╜
6batch_normalization_1/moving_variance/Initializer/onesConst*
valueB@*  А?*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:@
э
%batch_normalization_1/moving_varianceVarHandleOp*6
shared_name'%batch_normalization_1/moving_variance*8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container *
shape:@*
dtype0*
_output_shapes
: 
Ы
Fbatch_normalization_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
ц
,batch_normalization_1/moving_variance/AssignAssignVariableOp%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0
╒
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0
z
!batch_normalization_1/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_1/cond/switch_tIdentity#batch_normalization_1/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_1/cond/switch_fIdentity!batch_normalization_1/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_1/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
Ш
)batch_normalization_1/cond/ReadVariableOpReadVariableOp2batch_normalization_1/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
:@
╬
0batch_normalization_1/cond/ReadVariableOp/SwitchSwitchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: : 
Ь
+batch_normalization_1/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_1/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
:@
╬
2batch_normalization_1/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_1/beta"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: : 
Й
 batch_normalization_1/cond/ConstConst$^batch_normalization_1/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
Л
"batch_normalization_1/cond/Const_1Const$^batch_normalization_1/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
Ы
)batch_normalization_1/cond/FusedBatchNormFusedBatchNorm2batch_normalization_1/cond/FusedBatchNorm/Switch:1)batch_normalization_1/cond/ReadVariableOp+batch_normalization_1/cond/ReadVariableOp_1 batch_normalization_1/cond/Const"batch_normalization_1/cond/Const_1*
data_formatNHWC*G
_output_shapes5
3:         uu@:@:@:@:@*
is_training(*
epsilon%oГ:*
T0
ь
0batch_normalization_1/cond/FusedBatchNorm/SwitchSwitchactivation_1/Relu"batch_normalization_1/cond/pred_id*
T0*$
_class
loc:@activation_1/Relu*J
_output_shapes8
6:         uu@:         uu@
Ъ
+batch_normalization_1/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_1/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
:@
╨
2batch_normalization_1/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: : 
Ъ
+batch_normalization_1/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_1/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes
:@
╬
2batch_normalization_1/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_1/beta"batch_normalization_1/cond/pred_id*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: : *
T0
╕
:batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
:@
ы
Abatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_1/moving_mean"batch_normalization_1/cond/pred_id*
_output_shapes
: : *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
╝
<batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:@
ї
Cbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_1/moving_variance"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: : 
╙
+batch_normalization_1/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_1/cond/FusedBatchNorm_1/Switch+batch_normalization_1/cond/ReadVariableOp_2+batch_normalization_1/cond/ReadVariableOp_3:batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1*
data_formatNHWC*G
_output_shapes5
3:         uu@:@:@:@:@*
is_training( *
epsilon%oГ:*
T0
ю
2batch_normalization_1/cond/FusedBatchNorm_1/SwitchSwitchactivation_1/Relu"batch_normalization_1/cond/pred_id*$
_class
loc:@activation_1/Relu*J
_output_shapes8
6:         uu@:         uu@*
T0
╞
 batch_normalization_1/cond/MergeMerge+batch_normalization_1/cond/FusedBatchNorm_1)batch_normalization_1/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:         uu@: 
╖
"batch_normalization_1/cond/Merge_1Merge-batch_normalization_1/cond/FusedBatchNorm_1:1+batch_normalization_1/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:@: 
╖
"batch_normalization_1/cond/Merge_2Merge-batch_normalization_1/cond/FusedBatchNorm_1:2+batch_normalization_1/cond/FusedBatchNorm:2*
_output_shapes

:@: *
T0*
N
|
#batch_normalization_1/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_1/cond_1/switch_tIdentity%batch_normalization_1/cond_1/Switch:1*
T0
*
_output_shapes
: 
w
%batch_normalization_1/cond_1/switch_fIdentity#batch_normalization_1/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_1/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
П
"batch_normalization_1/cond_1/ConstConst&^batch_normalization_1/cond_1/switch_t*
valueB
 *дp}?*
dtype0*
_output_shapes
: 
С
$batch_normalization_1/cond_1/Const_1Const&^batch_normalization_1/cond_1/switch_f*
valueB
 *  А?*
dtype0*
_output_shapes
: 
б
"batch_normalization_1/cond_1/MergeMerge$batch_normalization_1/cond_1/Const_1"batch_normalization_1/cond_1/Const*
N*
_output_shapes
: : *
T0
ж
+batch_normalization_1/AssignMovingAvg/sub/xConst*
valueB
 *  А?*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
: 
╪
)batch_normalization_1/AssignMovingAvg/subSub+batch_normalization_1/AssignMovingAvg/sub/x"batch_normalization_1/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: 
Т
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:@
ч
+batch_normalization_1/AssignMovingAvg/sub_1Sub4batch_normalization_1/AssignMovingAvg/ReadVariableOp"batch_normalization_1/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:@
у
)batch_normalization_1/AssignMovingAvg/mulMul+batch_normalization_1/AssignMovingAvg/sub_1)batch_normalization_1/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:@*
T0
с
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0
Ж
6batch_normalization_1/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_1/moving_mean:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:@
м
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*
valueB
 *  А?*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
: 
р
+batch_normalization_1/AssignMovingAvg_1/subSub-batch_normalization_1/AssignMovingAvg_1/sub/x"batch_normalization_1/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: 
Ш
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:@
я
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp"batch_normalization_1/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:@
э
+batch_normalization_1/AssignMovingAvg_1/mulMul-batch_normalization_1/AssignMovingAvg_1/sub_1+batch_normalization_1/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:@
э
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0
Т
8batch_normalization_1/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_1/moving_variance<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*
_output_shapes
:@*8
_class.
,*loc:@batch_normalization_1/moving_variance
╤
max_pooling2d_1/MaxPoolMaxPool batch_normalization_1/cond/Merge*
ksize
*
paddingVALID*/
_output_shapes
:         ::@*
T0*
strides
*
data_formatNHWC
н
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   А   *"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
:
Ч
.conv2d_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *├╨╜*"
_class
loc:@conv2d_2/kernel*
dtype0
Ч
.conv2d_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *├╨=*"
_class
loc:@conv2d_2/kernel
ў
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:@А*

seed *
T0*"
_class
loc:@conv2d_2/kernel*
seed2 
┌
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
ї
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*'
_output_shapes
:@А*
T0*"
_class
loc:@conv2d_2/kernel
ч
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*'
_output_shapes
:@А
╕
conv2d_2/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_nameconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
	container *
shape:@А
o
0conv2d_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
Ш
conv2d_2/kernel/AssignAssignVariableOpconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_2/kernel*
dtype0
а
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*'
_output_shapes
:@А*"
_class
loc:@conv2d_2/kernel
Р
conv2d_2/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*
valueBА*    * 
_class
loc:@conv2d_2/bias
ж
conv2d_2/biasVarHandleOp*
shape:А*
dtype0*
_output_shapes
: *
shared_nameconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
	container 
k
.conv2d_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/bias*
_output_shapes
: 
З
conv2d_2/bias/AssignAssignVariableOpconv2d_2/biasconv2d_2/bias/Initializer/zeros* 
_class
loc:@conv2d_2/bias*
dtype0
О
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes	
:А* 
_class
loc:@conv2d_2/bias
g
conv2d_2/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
w
conv2d_2/Conv2D/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*'
_output_shapes
:@А
Х
conv2d_2/Conv2DConv2Dmax_pooling2d_1/MaxPoolconv2d_2/Conv2D/ReadVariableOp*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:         66А*
	dilations
*
T0
j
conv2d_2/BiasAdd/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:А*
dtype0
Я
conv2d_2/BiasAddBiasAddconv2d_2/Conv2Dconv2d_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:         66А
f
activation_2/ReluReluconv2d_2/BiasAdd*0
_output_shapes
:         66А*
T0
л
,batch_normalization_2/gamma/Initializer/onesConst*
_output_shapes	
:А*
valueBА*  А?*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0
╨
batch_normalization_2/gammaVarHandleOp*
	container *
shape:А*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_2/gamma*.
_class$
" loc:@batch_normalization_2/gamma
З
<batch_normalization_2/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
╛
"batch_normalization_2/gamma/AssignAssignVariableOpbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*
dtype0*.
_class$
" loc:@batch_normalization_2/gamma
╕
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes	
:А
к
,batch_normalization_2/beta/Initializer/zerosConst*
valueBА*    *-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes	
:А
═
batch_normalization_2/betaVarHandleOp*+
shared_namebatch_normalization_2/beta*-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:А*
dtype0*
_output_shapes
: 
Е
;batch_normalization_2/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
╗
!batch_normalization_2/beta/AssignAssignVariableOpbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_2/beta*
dtype0
╡
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
dtype0*
_output_shapes	
:А*-
_class#
!loc:@batch_normalization_2/beta
╕
3batch_normalization_2/moving_mean/Initializer/zerosConst*
valueBА*    *4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes	
:А
т
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *2
shared_name#!batch_normalization_2/moving_mean*4
_class*
(&loc:@batch_normalization_2/moving_mean*
	container *
shape:А*
dtype0
У
Bbatch_normalization_2/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
╫
(batch_normalization_2/moving_mean/AssignAssignVariableOp!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros*
dtype0*4
_class*
(&loc:@batch_normalization_2/moving_mean
╩
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes	
:А
┐
6batch_normalization_2/moving_variance/Initializer/onesConst*
_output_shapes	
:А*
valueBА*  А?*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0
ю
%batch_normalization_2/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_2/moving_variance*8
_class.
,*loc:@batch_normalization_2/moving_variance*
	container *
shape:А
Ы
Fbatch_normalization_2/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
ц
,batch_normalization_2/moving_variance/AssignAssignVariableOp%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*
dtype0*8
_class.
,*loc:@batch_normalization_2/moving_variance
╓
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes	
:А
z
!batch_normalization_2/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_2/cond/switch_tIdentity#batch_normalization_2/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_2/cond/switch_fIdentity!batch_normalization_2/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_2/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
Щ
)batch_normalization_2/cond/ReadVariableOpReadVariableOp2batch_normalization_2/cond/ReadVariableOp/Switch:1*
_output_shapes	
:А*
dtype0
╬
0batch_normalization_2/cond/ReadVariableOp/SwitchSwitchbatch_normalization_2/gamma"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: : 
Э
+batch_normalization_2/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_2/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes	
:А
╬
2batch_normalization_2/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_2/beta"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: : 
Й
 batch_normalization_2/cond/ConstConst$^batch_normalization_2/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
Л
"batch_normalization_2/cond/Const_1Const$^batch_normalization_2/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
а
)batch_normalization_2/cond/FusedBatchNormFusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm/Switch:1)batch_normalization_2/cond/ReadVariableOp+batch_normalization_2/cond/ReadVariableOp_1 batch_normalization_2/cond/Const"batch_normalization_2/cond/Const_1*
T0*
data_formatNHWC*L
_output_shapes:
8:         66А:А:А:А:А*
is_training(*
epsilon%oГ:
ю
0batch_normalization_2/cond/FusedBatchNorm/SwitchSwitchactivation_2/Relu"batch_normalization_2/cond/pred_id*$
_class
loc:@activation_2/Relu*L
_output_shapes:
8:         66А:         66А*
T0
Ы
+batch_normalization_2/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_2/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:А
╨
2batch_normalization_2/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_2/gamma"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: : 
Ы
+batch_normalization_2/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_2/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes	
:А
╬
2batch_normalization_2/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_2/beta"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: : 
╣
:batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
_output_shapes	
:А*
dtype0
ы
Abatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_2/moving_mean"batch_normalization_2/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: : 
╜
<batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:А
ї
Cbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_2/moving_variance"batch_normalization_2/cond/pred_id*
_output_shapes
: : *
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
╪
+batch_normalization_2/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm_1/Switch+batch_normalization_2/cond/ReadVariableOp_2+batch_normalization_2/cond/ReadVariableOp_3:batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1*
T0*
data_formatNHWC*L
_output_shapes:
8:         66А:А:А:А:А*
is_training( *
epsilon%oГ:
Ё
2batch_normalization_2/cond/FusedBatchNorm_1/SwitchSwitchactivation_2/Relu"batch_normalization_2/cond/pred_id*
T0*$
_class
loc:@activation_2/Relu*L
_output_shapes:
8:         66А:         66А
╟
 batch_normalization_2/cond/MergeMerge+batch_normalization_2/cond/FusedBatchNorm_1)batch_normalization_2/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:         66А: 
╕
"batch_normalization_2/cond/Merge_1Merge-batch_normalization_2/cond/FusedBatchNorm_1:1+batch_normalization_2/cond/FusedBatchNorm:1*
N*
_output_shapes
	:А: *
T0
╕
"batch_normalization_2/cond/Merge_2Merge-batch_normalization_2/cond/FusedBatchNorm_1:2+batch_normalization_2/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes
	:А: 
|
#batch_normalization_2/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

y
%batch_normalization_2/cond_1/switch_tIdentity%batch_normalization_2/cond_1/Switch:1*
T0
*
_output_shapes
: 
w
%batch_normalization_2/cond_1/switch_fIdentity#batch_normalization_2/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_2/cond_1/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

П
"batch_normalization_2/cond_1/ConstConst&^batch_normalization_2/cond_1/switch_t*
valueB
 *дp}?*
dtype0*
_output_shapes
: 
С
$batch_normalization_2/cond_1/Const_1Const&^batch_normalization_2/cond_1/switch_f*
valueB
 *  А?*
dtype0*
_output_shapes
: 
б
"batch_normalization_2/cond_1/MergeMerge$batch_normalization_2/cond_1/Const_1"batch_normalization_2/cond_1/Const*
N*
_output_shapes
: : *
T0
ж
+batch_normalization_2/AssignMovingAvg/sub/xConst*
valueB
 *  А?*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
: 
╪
)batch_normalization_2/AssignMovingAvg/subSub+batch_normalization_2/AssignMovingAvg/sub/x"batch_normalization_2/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: 
У
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:А*
dtype0
ш
+batch_normalization_2/AssignMovingAvg/sub_1Sub4batch_normalization_2/AssignMovingAvg/ReadVariableOp"batch_normalization_2/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes	
:А
ф
)batch_normalization_2/AssignMovingAvg/mulMul+batch_normalization_2/AssignMovingAvg/sub_1)batch_normalization_2/AssignMovingAvg/sub*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes	
:А
с
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_2/moving_mean)batch_normalization_2/AssignMovingAvg/mul*
dtype0*4
_class*
(&loc:@batch_normalization_2/moving_mean
З
6batch_normalization_2/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_2/moving_mean:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes	
:А
м
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*
valueB
 *  А?*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes
: 
р
+batch_normalization_2/AssignMovingAvg_1/subSub-batch_normalization_2/AssignMovingAvg_1/sub/x"batch_normalization_2/cond_1/Merge*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: *
T0
Щ
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes	
:А
Ё
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp"batch_normalization_2/cond/Merge_2*
_output_shapes	
:А*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
ю
+batch_normalization_2/AssignMovingAvg_1/mulMul-batch_normalization_2/AssignMovingAvg_1/sub_1+batch_normalization_2/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:А
э
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_2/moving_variance+batch_normalization_2/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0
У
8batch_normalization_2/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_2/moving_variance<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes	
:А
╥
max_pooling2d_2/MaxPoolMaxPool batch_normalization_2/cond/Merge*0
_output_shapes
:         А*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID
н
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      А      *"
_class
loc:@conv2d_3/kernel
Ч
.conv2d_3/kernel/Initializer/random_uniform/minConst*
valueB
 *лк*╜*"
_class
loc:@conv2d_3/kernel*
dtype0*
_output_shapes
: 
Ч
.conv2d_3/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *лк*=*"
_class
loc:@conv2d_3/kernel*
dtype0
°
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*(
_output_shapes
:АА*

seed *
T0*"
_class
loc:@conv2d_3/kernel
┌
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
Ў
.conv2d_3/kernel/Initializer/random_uniform/mulMul8conv2d_3/kernel/Initializer/random_uniform/RandomUniform.conv2d_3/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_3/kernel*(
_output_shapes
:АА
ш
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_3/kernel*(
_output_shapes
:АА*
T0
╣
conv2d_3/kernelVarHandleOp*
	container *
shape:АА*
dtype0*
_output_shapes
: * 
shared_nameconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel
o
0conv2d_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 
Ш
conv2d_3/kernel/AssignAssignVariableOpconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_3/kernel*
dtype0
б
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*
dtype0*(
_output_shapes
:АА*"
_class
loc:@conv2d_3/kernel
Р
conv2d_3/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*
valueBА*    * 
_class
loc:@conv2d_3/bias
ж
conv2d_3/biasVarHandleOp*
shared_nameconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
k
.conv2d_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/bias*
_output_shapes
: 
З
conv2d_3/bias/AssignAssignVariableOpconv2d_3/biasconv2d_3/bias/Initializer/zeros* 
_class
loc:@conv2d_3/bias*
dtype0
О
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes	
:А
g
conv2d_3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
x
conv2d_3/Conv2D/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:АА*
dtype0
Х
conv2d_3/Conv2DConv2Dmax_pooling2d_2/MaxPoolconv2d_3/Conv2D/ReadVariableOp*0
_output_shapes
:         А*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
j
conv2d_3/BiasAdd/ReadVariableOpReadVariableOpconv2d_3/bias*
dtype0*
_output_shapes	
:А
Я
conv2d_3/BiasAddBiasAddconv2d_3/Conv2Dconv2d_3/BiasAdd/ReadVariableOp*
data_formatNHWC*0
_output_shapes
:         А*
T0
f
activation_3/ReluReluconv2d_3/BiasAdd*
T0*0
_output_shapes
:         А
л
,batch_normalization_3/gamma/Initializer/onesConst*
valueBА*  А?*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes	
:А
╨
batch_normalization_3/gammaVarHandleOp*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_3/gamma*.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:А
З
<batch_normalization_3/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
╛
"batch_normalization_3/gamma/AssignAssignVariableOpbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0
╕
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes	
:А
к
,batch_normalization_3/beta/Initializer/zerosConst*
valueBА*    *-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes	
:А
═
batch_normalization_3/betaVarHandleOp*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_3/beta*-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:А
Е
;batch_normalization_3/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
╗
!batch_normalization_3/beta/AssignAssignVariableOpbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_3/beta*
dtype0
╡
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes	
:А
╕
3batch_normalization_3/moving_mean/Initializer/zerosConst*
valueBА*    *4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes	
:А
т
!batch_normalization_3/moving_meanVarHandleOp*4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container *
shape:А*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_3/moving_mean
У
Bbatch_normalization_3/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
╫
(batch_normalization_3/moving_mean/AssignAssignVariableOp!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
╩
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes	
:А
┐
6batch_normalization_3/moving_variance/Initializer/onesConst*
_output_shapes	
:А*
valueBА*  А?*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0
ю
%batch_normalization_3/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_3/moving_variance*8
_class.
,*loc:@batch_normalization_3/moving_variance*
	container *
shape:А
Ы
Fbatch_normalization_3/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
ц
,batch_normalization_3/moving_variance/AssignAssignVariableOp%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0
╓
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes	
:А
z
!batch_normalization_3/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
_output_shapes
: *
T0

e
"batch_normalization_3/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
Щ
)batch_normalization_3/cond/ReadVariableOpReadVariableOp2batch_normalization_3/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes	
:А
╬
0batch_normalization_3/cond/ReadVariableOp/SwitchSwitchbatch_normalization_3/gamma"batch_normalization_3/cond/pred_id*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
: : *
T0
Э
+batch_normalization_3/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_3/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes	
:А
╬
2batch_normalization_3/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_3/beta"batch_normalization_3/cond/pred_id*
_output_shapes
: : *
T0*-
_class#
!loc:@batch_normalization_3/beta
Й
 batch_normalization_3/cond/ConstConst$^batch_normalization_3/cond/switch_t*
_output_shapes
: *
valueB *
dtype0
Л
"batch_normalization_3/cond/Const_1Const$^batch_normalization_3/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
а
)batch_normalization_3/cond/FusedBatchNormFusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm/Switch:1)batch_normalization_3/cond/ReadVariableOp+batch_normalization_3/cond/ReadVariableOp_1 batch_normalization_3/cond/Const"batch_normalization_3/cond/Const_1*
epsilon%oГ:*
T0*
data_formatNHWC*L
_output_shapes:
8:         А:А:А:А:А*
is_training(
ю
0batch_normalization_3/cond/FusedBatchNorm/SwitchSwitchactivation_3/Relu"batch_normalization_3/cond/pred_id*
T0*$
_class
loc:@activation_3/Relu*L
_output_shapes:
8:         А:         А
Ы
+batch_normalization_3/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_3/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:А
╨
2batch_normalization_3/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_3/gamma"batch_normalization_3/cond/pred_id*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
: : *
T0
Ы
+batch_normalization_3/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_3/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes	
:А
╬
2batch_normalization_3/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_3/beta"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
: : 
╣
:batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:А
ы
Abatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_3/moving_mean"batch_normalization_3/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: : 
╜
<batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
_output_shapes	
:А*
dtype0
ї
Cbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_3/moving_variance"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
: : 
╪
+batch_normalization_3/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm_1/Switch+batch_normalization_3/cond/ReadVariableOp_2+batch_normalization_3/cond/ReadVariableOp_3:batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1*
epsilon%oГ:*
T0*
data_formatNHWC*L
_output_shapes:
8:         А:А:А:А:А*
is_training( 
Ё
2batch_normalization_3/cond/FusedBatchNorm_1/SwitchSwitchactivation_3/Relu"batch_normalization_3/cond/pred_id*
T0*$
_class
loc:@activation_3/Relu*L
_output_shapes:
8:         А:         А
╟
 batch_normalization_3/cond/MergeMerge+batch_normalization_3/cond/FusedBatchNorm_1)batch_normalization_3/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:         А: 
╕
"batch_normalization_3/cond/Merge_1Merge-batch_normalization_3/cond/FusedBatchNorm_1:1+batch_normalization_3/cond/FusedBatchNorm:1*
N*
_output_shapes
	:А: *
T0
╕
"batch_normalization_3/cond/Merge_2Merge-batch_normalization_3/cond/FusedBatchNorm_1:2+batch_normalization_3/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes
	:А: 
|
#batch_normalization_3/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

y
%batch_normalization_3/cond_1/switch_tIdentity%batch_normalization_3/cond_1/Switch:1*
T0
*
_output_shapes
: 
w
%batch_normalization_3/cond_1/switch_fIdentity#batch_normalization_3/cond_1/Switch*
_output_shapes
: *
T0

g
$batch_normalization_3/cond_1/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

П
"batch_normalization_3/cond_1/ConstConst&^batch_normalization_3/cond_1/switch_t*
valueB
 *дp}?*
dtype0*
_output_shapes
: 
С
$batch_normalization_3/cond_1/Const_1Const&^batch_normalization_3/cond_1/switch_f*
dtype0*
_output_shapes
: *
valueB
 *  А?
б
"batch_normalization_3/cond_1/MergeMerge$batch_normalization_3/cond_1/Const_1"batch_normalization_3/cond_1/Const*
N*
_output_shapes
: : *
T0
ж
+batch_normalization_3/AssignMovingAvg/sub/xConst*
valueB
 *  А?*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
: 
╪
)batch_normalization_3/AssignMovingAvg/subSub+batch_normalization_3/AssignMovingAvg/sub/x"batch_normalization_3/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: 
У
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes	
:А
ш
+batch_normalization_3/AssignMovingAvg/sub_1Sub4batch_normalization_3/AssignMovingAvg/ReadVariableOp"batch_normalization_3/cond/Merge_1*
_output_shapes	
:А*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
ф
)batch_normalization_3/AssignMovingAvg/mulMul+batch_normalization_3/AssignMovingAvg/sub_1)batch_normalization_3/AssignMovingAvg/sub*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:А
с
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_3/moving_mean)batch_normalization_3/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
З
6batch_normalization_3/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_3/moving_mean:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes	
:А
м
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*
valueB
 *  А?*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: 
р
+batch_normalization_3/AssignMovingAvg_1/subSub-batch_normalization_3/AssignMovingAvg_1/sub/x"batch_normalization_3/cond_1/Merge*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
Щ
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes	
:А
Ё
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp"batch_normalization_3/cond/Merge_2*
_output_shapes	
:А*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
ю
+batch_normalization_3/AssignMovingAvg_1/mulMul-batch_normalization_3/AssignMovingAvg_1/sub_1+batch_normalization_3/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:А
э
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_3/moving_variance+batch_normalization_3/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0
У
8batch_normalization_3/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_3/moving_variance<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes	
:А
╥
max_pooling2d_3/MaxPoolMaxPool batch_normalization_3/cond/Merge*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:         А
н
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
:
Ч
.conv2d_4/kernel/Initializer/random_uniform/minConst*
valueB
 *я[ё╝*"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
: 
Ч
.conv2d_4/kernel/Initializer/random_uniform/maxConst*
valueB
 *я[ё<*"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
: 
°
8conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_4/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*(
_output_shapes
:АА*

seed *
T0*"
_class
loc:@conv2d_4/kernel
┌
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_4/kernel
Ў
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*(
_output_shapes
:АА*
T0*"
_class
loc:@conv2d_4/kernel
ш
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*(
_output_shapes
:АА
╣
conv2d_4/kernelVarHandleOp*
_output_shapes
: * 
shared_nameconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
	container *
shape:АА*
dtype0
o
0conv2d_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/kernel*
_output_shapes
: 
Ш
conv2d_4/kernel/AssignAssignVariableOpconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_4/kernel*
dtype0
б
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
dtype0*(
_output_shapes
:АА
Р
conv2d_4/bias/Initializer/zerosConst*
_output_shapes	
:А*
valueBА*    * 
_class
loc:@conv2d_4/bias*
dtype0
ж
conv2d_4/biasVarHandleOp*
	container *
shape:А*
dtype0*
_output_shapes
: *
shared_nameconv2d_4/bias* 
_class
loc:@conv2d_4/bias
k
.conv2d_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/bias*
_output_shapes
: 
З
conv2d_4/bias/AssignAssignVariableOpconv2d_4/biasconv2d_4/bias/Initializer/zeros* 
_class
loc:@conv2d_4/bias*
dtype0
О
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes	
:А
g
conv2d_4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
x
conv2d_4/Conv2D/ReadVariableOpReadVariableOpconv2d_4/kernel*
dtype0*(
_output_shapes
:АА
Х
conv2d_4/Conv2DConv2Dmax_pooling2d_3/MaxPoolconv2d_4/Conv2D/ReadVariableOp*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:         

А
j
conv2d_4/BiasAdd/ReadVariableOpReadVariableOpconv2d_4/bias*
dtype0*
_output_shapes	
:А
Я
conv2d_4/BiasAddBiasAddconv2d_4/Conv2Dconv2d_4/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:         

А
f
activation_4/ReluReluconv2d_4/BiasAdd*0
_output_shapes
:         

А*
T0
л
,batch_normalization_4/gamma/Initializer/onesConst*
valueBА*  А?*.
_class$
" loc:@batch_normalization_4/gamma*
dtype0*
_output_shapes	
:А
╨
batch_normalization_4/gammaVarHandleOp*,
shared_namebatch_normalization_4/gamma*.
_class$
" loc:@batch_normalization_4/gamma*
	container *
shape:А*
dtype0*
_output_shapes
: 
З
<batch_normalization_4/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
╛
"batch_normalization_4/gamma/AssignAssignVariableOpbatch_normalization_4/gamma,batch_normalization_4/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_4/gamma*
dtype0
╕
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes	
:А*.
_class$
" loc:@batch_normalization_4/gamma
к
,batch_normalization_4/beta/Initializer/zerosConst*
valueBА*    *-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes	
:А
═
batch_normalization_4/betaVarHandleOp*-
_class#
!loc:@batch_normalization_4/beta*
	container *
shape:А*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_4/beta
Е
;batch_normalization_4/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
╗
!batch_normalization_4/beta/AssignAssignVariableOpbatch_normalization_4/beta,batch_normalization_4/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_4/beta*
dtype0
╡
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:А*-
_class#
!loc:@batch_normalization_4/beta*
dtype0
╕
3batch_normalization_4/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*
valueBА*    *4
_class*
(&loc:@batch_normalization_4/moving_mean
т
!batch_normalization_4/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_4/moving_mean*4
_class*
(&loc:@batch_normalization_4/moving_mean*
	container *
shape:А
У
Bbatch_normalization_4/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
╫
(batch_normalization_4/moving_mean/AssignAssignVariableOp!batch_normalization_4/moving_mean3batch_normalization_4/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0
╩
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes	
:А
┐
6batch_normalization_4/moving_variance/Initializer/onesConst*
_output_shapes	
:А*
valueBА*  А?*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0
ю
%batch_normalization_4/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_4/moving_variance*8
_class.
,*loc:@batch_normalization_4/moving_variance*
	container *
shape:А
Ы
Fbatch_normalization_4/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
ц
,batch_normalization_4/moving_variance/AssignAssignVariableOp%batch_normalization_4/moving_variance6batch_normalization_4/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0
╓
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes	
:А
z
!batch_normalization_4/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_4/cond/switch_tIdentity#batch_normalization_4/cond/Switch:1*
_output_shapes
: *
T0

s
#batch_normalization_4/cond/switch_fIdentity!batch_normalization_4/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_4/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

Щ
)batch_normalization_4/cond/ReadVariableOpReadVariableOp2batch_normalization_4/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes	
:А
╬
0batch_normalization_4/cond/ReadVariableOp/SwitchSwitchbatch_normalization_4/gamma"batch_normalization_4/cond/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_4/gamma
Э
+batch_normalization_4/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_4/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes	
:А
╬
2batch_normalization_4/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_4/beta"batch_normalization_4/cond/pred_id*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: : *
T0
Й
 batch_normalization_4/cond/ConstConst$^batch_normalization_4/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
Л
"batch_normalization_4/cond/Const_1Const$^batch_normalization_4/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
а
)batch_normalization_4/cond/FusedBatchNormFusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm/Switch:1)batch_normalization_4/cond/ReadVariableOp+batch_normalization_4/cond/ReadVariableOp_1 batch_normalization_4/cond/Const"batch_normalization_4/cond/Const_1*
data_formatNHWC*L
_output_shapes:
8:         

А:А:А:А:А*
is_training(*
epsilon%oГ:*
T0
ю
0batch_normalization_4/cond/FusedBatchNorm/SwitchSwitchactivation_4/Relu"batch_normalization_4/cond/pred_id*
T0*$
_class
loc:@activation_4/Relu*L
_output_shapes:
8:         

А:         

А
Ы
+batch_normalization_4/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_4/cond/ReadVariableOp_2/Switch*
_output_shapes	
:А*
dtype0
╨
2batch_normalization_4/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_4/gamma"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
: : 
Ы
+batch_normalization_4/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_4/cond/ReadVariableOp_3/Switch*
_output_shapes	
:А*
dtype0
╬
2batch_normalization_4/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_4/beta"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: : 
╣
:batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:А
ы
Abatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_4/moving_mean"batch_normalization_4/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
: : 
╜
<batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:А
ї
Cbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_4/moving_variance"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: : 
╪
+batch_normalization_4/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm_1/Switch+batch_normalization_4/cond/ReadVariableOp_2+batch_normalization_4/cond/ReadVariableOp_3:batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1*
T0*
data_formatNHWC*L
_output_shapes:
8:         

А:А:А:А:А*
is_training( *
epsilon%oГ:
Ё
2batch_normalization_4/cond/FusedBatchNorm_1/SwitchSwitchactivation_4/Relu"batch_normalization_4/cond/pred_id*$
_class
loc:@activation_4/Relu*L
_output_shapes:
8:         

А:         

А*
T0
╟
 batch_normalization_4/cond/MergeMerge+batch_normalization_4/cond/FusedBatchNorm_1)batch_normalization_4/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:         

А: 
╕
"batch_normalization_4/cond/Merge_1Merge-batch_normalization_4/cond/FusedBatchNorm_1:1+batch_normalization_4/cond/FusedBatchNorm:1*
N*
_output_shapes
	:А: *
T0
╕
"batch_normalization_4/cond/Merge_2Merge-batch_normalization_4/cond/FusedBatchNorm_1:2+batch_normalization_4/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes
	:А: 
|
#batch_normalization_4/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_4/cond_1/switch_tIdentity%batch_normalization_4/cond_1/Switch:1*
T0
*
_output_shapes
: 
w
%batch_normalization_4/cond_1/switch_fIdentity#batch_normalization_4/cond_1/Switch*
_output_shapes
: *
T0

g
$batch_normalization_4/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
П
"batch_normalization_4/cond_1/ConstConst&^batch_normalization_4/cond_1/switch_t*
valueB
 *дp}?*
dtype0*
_output_shapes
: 
С
$batch_normalization_4/cond_1/Const_1Const&^batch_normalization_4/cond_1/switch_f*
valueB
 *  А?*
dtype0*
_output_shapes
: 
б
"batch_normalization_4/cond_1/MergeMerge$batch_normalization_4/cond_1/Const_1"batch_normalization_4/cond_1/Const*
T0*
N*
_output_shapes
: : 
ж
+batch_normalization_4/AssignMovingAvg/sub/xConst*
valueB
 *  А?*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes
: 
╪
)batch_normalization_4/AssignMovingAvg/subSub+batch_normalization_4/AssignMovingAvg/sub/x"batch_normalization_4/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
: 
У
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes	
:А
ш
+batch_normalization_4/AssignMovingAvg/sub_1Sub4batch_normalization_4/AssignMovingAvg/ReadVariableOp"batch_normalization_4/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes	
:А
ф
)batch_normalization_4/AssignMovingAvg/mulMul+batch_normalization_4/AssignMovingAvg/sub_1)batch_normalization_4/AssignMovingAvg/sub*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes	
:А
с
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_4/moving_mean)batch_normalization_4/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0
З
6batch_normalization_4/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_4/moving_mean:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp*
_output_shapes	
:А*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0
м
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
valueB
 *  А?*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0
р
+batch_normalization_4/AssignMovingAvg_1/subSub-batch_normalization_4/AssignMovingAvg_1/sub/x"batch_normalization_4/cond_1/Merge*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
Щ
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes	
:А
Ё
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp"batch_normalization_4/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes	
:А
ю
+batch_normalization_4/AssignMovingAvg_1/mulMul-batch_normalization_4/AssignMovingAvg_1/sub_1+batch_normalization_4/AssignMovingAvg_1/sub*
_output_shapes	
:А*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
э
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_4/moving_variance+batch_normalization_4/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0
У
8batch_normalization_4/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_4/moving_variance<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes	
:А
╥
max_pooling2d_4/MaxPoolMaxPool batch_normalization_4/cond/Merge*0
_output_shapes
:         А*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
d
flatten/ShapeShapemax_pooling2d_4/MaxPool*
T0*
out_type0*
_output_shapes
:
e
flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
б
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
b
flatten/Reshape/shape/1Const*
valueB :
         *
dtype0*
_output_shapes
: 
З
flatten/Reshape/shapePackflatten/strided_sliceflatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Л
flatten/ReshapeReshapemax_pooling2d_4/MaxPoolflatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:         Аd
Я
-dense/kernel/Initializer/random_uniform/shapeConst*
valueB" 2     *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:
С
+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *лкк╝*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
С
+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *лкк<*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
ч
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
АdА*

seed *
T0*
_class
loc:@dense/kernel*
seed2 *
dtype0
╬
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
т
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АdА*
T0*
_class
loc:@dense/kernel
╘
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
АdА
и
dense/kernelVarHandleOp*
shared_namedense/kernel*
_class
loc:@dense/kernel*
	container *
shape:
АdА*
dtype0*
_output_shapes
: 
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
М
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
dtype0
Р
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_class
loc:@dense/kernel*
dtype0* 
_output_shapes
:
АdА
Ц
,dense/bias/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:А*
_class
loc:@dense/bias*
dtype0
Ж
"dense/bias/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@dense/bias
═
dense/bias/Initializer/zerosFill,dense/bias/Initializer/zeros/shape_as_tensor"dense/bias/Initializer/zeros/Const*

index_type0*
_class
loc:@dense/bias*
_output_shapes	
:А*
T0
Э

dense/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_name
dense/bias*
_class
loc:@dense/bias*
	container *
shape:А
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
{
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
_class
loc:@dense/bias*
dtype0
Е
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes	
:А
j
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
АdА*
dtype0
Э
dense/MatMulMatMulflatten/Reshapedense/MatMul/ReadVariableOp*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
d
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes	
:А
О
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:         А
Z
dense/SigmoidSigmoiddense/BiasAdd*
T0*(
_output_shapes
:         А
й
2prediction/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *$
_class
loc:@prediction/kernel*
dtype0
Ы
0prediction/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *▒бЫ╜*$
_class
loc:@prediction/kernel*
dtype0
Ы
0prediction/kernel/Initializer/random_uniform/maxConst*
valueB
 *▒бЫ=*$
_class
loc:@prediction/kernel*
dtype0*
_output_shapes
: 
ї
:prediction/kernel/Initializer/random_uniform/RandomUniformRandomUniform2prediction/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А*

seed *
T0*$
_class
loc:@prediction/kernel*
seed2 
т
0prediction/kernel/Initializer/random_uniform/subSub0prediction/kernel/Initializer/random_uniform/max0prediction/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@prediction/kernel
ї
0prediction/kernel/Initializer/random_uniform/mulMul:prediction/kernel/Initializer/random_uniform/RandomUniform0prediction/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@prediction/kernel*
_output_shapes
:	А
ч
,prediction/kernel/Initializer/random_uniformAdd0prediction/kernel/Initializer/random_uniform/mul0prediction/kernel/Initializer/random_uniform/min*$
_class
loc:@prediction/kernel*
_output_shapes
:	А*
T0
╢
prediction/kernelVarHandleOp*
shape:	А*
dtype0*
_output_shapes
: *"
shared_nameprediction/kernel*$
_class
loc:@prediction/kernel*
	container 
s
2prediction/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpprediction/kernel*
_output_shapes
: 
а
prediction/kernel/AssignAssignVariableOpprediction/kernel,prediction/kernel/Initializer/random_uniform*
dtype0*$
_class
loc:@prediction/kernel
Ю
%prediction/kernel/Read/ReadVariableOpReadVariableOpprediction/kernel*$
_class
loc:@prediction/kernel*
dtype0*
_output_shapes
:	А
Т
!prediction/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@prediction/bias*
dtype0*
_output_shapes
:
л
prediction/biasVarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: * 
shared_nameprediction/bias*"
_class
loc:@prediction/bias
o
0prediction/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpprediction/bias*
_output_shapes
: 
П
prediction/bias/AssignAssignVariableOpprediction/bias!prediction/bias/Initializer/zeros*"
_class
loc:@prediction/bias*
dtype0
У
#prediction/bias/Read/ReadVariableOpReadVariableOpprediction/bias*"
_class
loc:@prediction/bias*
dtype0*
_output_shapes
:
s
 prediction/MatMul/ReadVariableOpReadVariableOpprediction/kernel*
dtype0*
_output_shapes
:	А
д
prediction/MatMulMatMuldense/Sigmoid prediction/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
m
!prediction/BiasAdd/ReadVariableOpReadVariableOpprediction/bias*
dtype0*
_output_shapes
:
Ь
prediction/BiasAddBiasAddprediction/MatMul!prediction/BiasAdd/ReadVariableOp*
data_formatNHWC*'
_output_shapes
:         *
T0
c
prediction/SoftmaxSoftmaxprediction/BiasAdd*
T0*'
_output_shapes
:         
l
PlaceholderPlaceholder*
dtype0*&
_output_shapes
:

 *
shape:

 
M
AssignVariableOpAssignVariableOpconv2d/kernelPlaceholder*
dtype0
w
ReadVariableOpReadVariableOpconv2d/kernel^AssignVariableOp*
dtype0*&
_output_shapes
:

 
V
Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 
O
AssignVariableOp_1AssignVariableOpconv2d/biasPlaceholder_1*
dtype0
m
ReadVariableOp_1ReadVariableOpconv2d/bias^AssignVariableOp_1*
dtype0*
_output_shapes
: 
V
Placeholder_2Placeholder*
dtype0*
_output_shapes
: *
shape: 
]
AssignVariableOp_2AssignVariableOpbatch_normalization/gammaPlaceholder_2*
dtype0
{
ReadVariableOp_2ReadVariableOpbatch_normalization/gamma^AssignVariableOp_2*
dtype0*
_output_shapes
: 
V
Placeholder_3Placeholder*
dtype0*
_output_shapes
: *
shape: 
\
AssignVariableOp_3AssignVariableOpbatch_normalization/betaPlaceholder_3*
dtype0
z
ReadVariableOp_3ReadVariableOpbatch_normalization/beta^AssignVariableOp_3*
dtype0*
_output_shapes
: 
V
Placeholder_4Placeholder*
_output_shapes
: *
shape: *
dtype0
c
AssignVariableOp_4AssignVariableOpbatch_normalization/moving_meanPlaceholder_4*
dtype0
Б
ReadVariableOp_4ReadVariableOpbatch_normalization/moving_mean^AssignVariableOp_4*
dtype0*
_output_shapes
: 
V
Placeholder_5Placeholder*
dtype0*
_output_shapes
: *
shape: 
g
AssignVariableOp_5AssignVariableOp#batch_normalization/moving_variancePlaceholder_5*
dtype0
Е
ReadVariableOp_5ReadVariableOp#batch_normalization/moving_variance^AssignVariableOp_5*
dtype0*
_output_shapes
: 
n
Placeholder_6Placeholder*
dtype0*&
_output_shapes
: @*
shape: @
S
AssignVariableOp_6AssignVariableOpconv2d_1/kernelPlaceholder_6*
dtype0
}
ReadVariableOp_6ReadVariableOpconv2d_1/kernel^AssignVariableOp_6*
dtype0*&
_output_shapes
: @
V
Placeholder_7Placeholder*
shape:@*
dtype0*
_output_shapes
:@
Q
AssignVariableOp_7AssignVariableOpconv2d_1/biasPlaceholder_7*
dtype0
o
ReadVariableOp_7ReadVariableOpconv2d_1/bias^AssignVariableOp_7*
dtype0*
_output_shapes
:@
V
Placeholder_8Placeholder*
shape:@*
dtype0*
_output_shapes
:@
_
AssignVariableOp_8AssignVariableOpbatch_normalization_1/gammaPlaceholder_8*
dtype0
}
ReadVariableOp_8ReadVariableOpbatch_normalization_1/gamma^AssignVariableOp_8*
_output_shapes
:@*
dtype0
V
Placeholder_9Placeholder*
shape:@*
dtype0*
_output_shapes
:@
^
AssignVariableOp_9AssignVariableOpbatch_normalization_1/betaPlaceholder_9*
dtype0
|
ReadVariableOp_9ReadVariableOpbatch_normalization_1/beta^AssignVariableOp_9*
dtype0*
_output_shapes
:@
W
Placeholder_10Placeholder*
shape:@*
dtype0*
_output_shapes
:@
g
AssignVariableOp_10AssignVariableOp!batch_normalization_1/moving_meanPlaceholder_10*
dtype0
Е
ReadVariableOp_10ReadVariableOp!batch_normalization_1/moving_mean^AssignVariableOp_10*
_output_shapes
:@*
dtype0
W
Placeholder_11Placeholder*
dtype0*
_output_shapes
:@*
shape:@
k
AssignVariableOp_11AssignVariableOp%batch_normalization_1/moving_variancePlaceholder_11*
dtype0
Й
ReadVariableOp_11ReadVariableOp%batch_normalization_1/moving_variance^AssignVariableOp_11*
dtype0*
_output_shapes
:@
q
Placeholder_12Placeholder*
dtype0*'
_output_shapes
:@А*
shape:@А
U
AssignVariableOp_12AssignVariableOpconv2d_2/kernelPlaceholder_12*
dtype0
А
ReadVariableOp_12ReadVariableOpconv2d_2/kernel^AssignVariableOp_12*
dtype0*'
_output_shapes
:@А
Y
Placeholder_13Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
S
AssignVariableOp_13AssignVariableOpconv2d_2/biasPlaceholder_13*
dtype0
r
ReadVariableOp_13ReadVariableOpconv2d_2/bias^AssignVariableOp_13*
_output_shapes	
:А*
dtype0
Y
Placeholder_14Placeholder*
_output_shapes	
:А*
shape:А*
dtype0
a
AssignVariableOp_14AssignVariableOpbatch_normalization_2/gammaPlaceholder_14*
dtype0
А
ReadVariableOp_14ReadVariableOpbatch_normalization_2/gamma^AssignVariableOp_14*
dtype0*
_output_shapes	
:А
Y
Placeholder_15Placeholder*
shape:А*
dtype0*
_output_shapes	
:А
`
AssignVariableOp_15AssignVariableOpbatch_normalization_2/betaPlaceholder_15*
dtype0

ReadVariableOp_15ReadVariableOpbatch_normalization_2/beta^AssignVariableOp_15*
_output_shapes	
:А*
dtype0
Y
Placeholder_16Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
g
AssignVariableOp_16AssignVariableOp!batch_normalization_2/moving_meanPlaceholder_16*
dtype0
Ж
ReadVariableOp_16ReadVariableOp!batch_normalization_2/moving_mean^AssignVariableOp_16*
dtype0*
_output_shapes	
:А
Y
Placeholder_17Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
k
AssignVariableOp_17AssignVariableOp%batch_normalization_2/moving_variancePlaceholder_17*
dtype0
К
ReadVariableOp_17ReadVariableOp%batch_normalization_2/moving_variance^AssignVariableOp_17*
_output_shapes	
:А*
dtype0
s
Placeholder_18Placeholder*
dtype0*(
_output_shapes
:АА*
shape:АА
U
AssignVariableOp_18AssignVariableOpconv2d_3/kernelPlaceholder_18*
dtype0
Б
ReadVariableOp_18ReadVariableOpconv2d_3/kernel^AssignVariableOp_18*(
_output_shapes
:АА*
dtype0
Y
Placeholder_19Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
S
AssignVariableOp_19AssignVariableOpconv2d_3/biasPlaceholder_19*
dtype0
r
ReadVariableOp_19ReadVariableOpconv2d_3/bias^AssignVariableOp_19*
dtype0*
_output_shapes	
:А
Y
Placeholder_20Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
a
AssignVariableOp_20AssignVariableOpbatch_normalization_3/gammaPlaceholder_20*
dtype0
А
ReadVariableOp_20ReadVariableOpbatch_normalization_3/gamma^AssignVariableOp_20*
dtype0*
_output_shapes	
:А
Y
Placeholder_21Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
`
AssignVariableOp_21AssignVariableOpbatch_normalization_3/betaPlaceholder_21*
dtype0

ReadVariableOp_21ReadVariableOpbatch_normalization_3/beta^AssignVariableOp_21*
dtype0*
_output_shapes	
:А
Y
Placeholder_22Placeholder*
shape:А*
dtype0*
_output_shapes	
:А
g
AssignVariableOp_22AssignVariableOp!batch_normalization_3/moving_meanPlaceholder_22*
dtype0
Ж
ReadVariableOp_22ReadVariableOp!batch_normalization_3/moving_mean^AssignVariableOp_22*
dtype0*
_output_shapes	
:А
Y
Placeholder_23Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
k
AssignVariableOp_23AssignVariableOp%batch_normalization_3/moving_variancePlaceholder_23*
dtype0
К
ReadVariableOp_23ReadVariableOp%batch_normalization_3/moving_variance^AssignVariableOp_23*
dtype0*
_output_shapes	
:А
s
Placeholder_24Placeholder*
shape:АА*
dtype0*(
_output_shapes
:АА
U
AssignVariableOp_24AssignVariableOpconv2d_4/kernelPlaceholder_24*
dtype0
Б
ReadVariableOp_24ReadVariableOpconv2d_4/kernel^AssignVariableOp_24*(
_output_shapes
:АА*
dtype0
Y
Placeholder_25Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
S
AssignVariableOp_25AssignVariableOpconv2d_4/biasPlaceholder_25*
dtype0
r
ReadVariableOp_25ReadVariableOpconv2d_4/bias^AssignVariableOp_25*
_output_shapes	
:А*
dtype0
Y
Placeholder_26Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
a
AssignVariableOp_26AssignVariableOpbatch_normalization_4/gammaPlaceholder_26*
dtype0
А
ReadVariableOp_26ReadVariableOpbatch_normalization_4/gamma^AssignVariableOp_26*
dtype0*
_output_shapes	
:А
Y
Placeholder_27Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
`
AssignVariableOp_27AssignVariableOpbatch_normalization_4/betaPlaceholder_27*
dtype0

ReadVariableOp_27ReadVariableOpbatch_normalization_4/beta^AssignVariableOp_27*
dtype0*
_output_shapes	
:А
Y
Placeholder_28Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
g
AssignVariableOp_28AssignVariableOp!batch_normalization_4/moving_meanPlaceholder_28*
dtype0
Ж
ReadVariableOp_28ReadVariableOp!batch_normalization_4/moving_mean^AssignVariableOp_28*
dtype0*
_output_shapes	
:А
Y
Placeholder_29Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
k
AssignVariableOp_29AssignVariableOp%batch_normalization_4/moving_variancePlaceholder_29*
dtype0
К
ReadVariableOp_29ReadVariableOp%batch_normalization_4/moving_variance^AssignVariableOp_29*
dtype0*
_output_shapes	
:А
c
Placeholder_30Placeholder*
shape:
АdА*
dtype0* 
_output_shapes
:
АdА
R
AssignVariableOp_30AssignVariableOpdense/kernelPlaceholder_30*
dtype0
v
ReadVariableOp_30ReadVariableOpdense/kernel^AssignVariableOp_30*
dtype0* 
_output_shapes
:
АdА
Y
Placeholder_31Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
P
AssignVariableOp_31AssignVariableOp
dense/biasPlaceholder_31*
dtype0
o
ReadVariableOp_31ReadVariableOp
dense/bias^AssignVariableOp_31*
dtype0*
_output_shapes	
:А
a
Placeholder_32Placeholder*
dtype0*
_output_shapes
:	А*
shape:	А
W
AssignVariableOp_32AssignVariableOpprediction/kernelPlaceholder_32*
dtype0
z
ReadVariableOp_32ReadVariableOpprediction/kernel^AssignVariableOp_32*
_output_shapes
:	А*
dtype0
W
Placeholder_33Placeholder*
dtype0*
_output_shapes
:*
shape:
U
AssignVariableOp_33AssignVariableOpprediction/biasPlaceholder_33*
dtype0
s
ReadVariableOp_33ReadVariableOpprediction/bias^AssignVariableOp_33*
dtype0*
_output_shapes
:
Q
VarIsInitializedOpVarIsInitializedOpprediction/bias*
_output_shapes
: 
P
VarIsInitializedOp_1VarIsInitializedOpdense/kernel*
_output_shapes
: 
i
VarIsInitializedOp_2VarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
Q
VarIsInitializedOp_3VarIsInitializedOpconv2d_3/bias*
_output_shapes
: 
c
VarIsInitializedOp_4VarIsInitializedOpbatch_normalization/moving_mean*
_output_shapes
: 
^
VarIsInitializedOp_5VarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
U
VarIsInitializedOp_6VarIsInitializedOpprediction/kernel*
_output_shapes
: 
S
VarIsInitializedOp_7VarIsInitializedOpconv2d_4/kernel*
_output_shapes
: 
e
VarIsInitializedOp_8VarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
i
VarIsInitializedOp_9VarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
`
VarIsInitializedOp_10VarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
T
VarIsInitializedOp_11VarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 
j
VarIsInitializedOp_12VarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
f
VarIsInitializedOp_13VarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
]
VarIsInitializedOp_14VarIsInitializedOpbatch_normalization/beta*
_output_shapes
: 
`
VarIsInitializedOp_15VarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
`
VarIsInitializedOp_16VarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
R
VarIsInitializedOp_17VarIsInitializedOpconv2d_1/bias*
_output_shapes
: 
R
VarIsInitializedOp_18VarIsInitializedOpconv2d_2/bias*
_output_shapes
: 
T
VarIsInitializedOp_19VarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
R
VarIsInitializedOp_20VarIsInitializedOpconv2d/kernel*
_output_shapes
: 
_
VarIsInitializedOp_21VarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
T
VarIsInitializedOp_22VarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
P
VarIsInitializedOp_23VarIsInitializedOpconv2d/bias*
_output_shapes
: 
^
VarIsInitializedOp_24VarIsInitializedOpbatch_normalization/gamma*
_output_shapes
: 
_
VarIsInitializedOp_25VarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
f
VarIsInitializedOp_26VarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
f
VarIsInitializedOp_27VarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
`
VarIsInitializedOp_28VarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
O
VarIsInitializedOp_29VarIsInitializedOp
dense/bias*
_output_shapes
: 
_
VarIsInitializedOp_30VarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
j
VarIsInitializedOp_31VarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
R
VarIsInitializedOp_32VarIsInitializedOpconv2d_4/bias*
_output_shapes
: 
h
VarIsInitializedOp_33VarIsInitializedOp#batch_normalization/moving_variance*
_output_shapes
: 
¤
initNoOp ^batch_normalization/beta/Assign!^batch_normalization/gamma/Assign'^batch_normalization/moving_mean/Assign+^batch_normalization/moving_variance/Assign"^batch_normalization_1/beta/Assign#^batch_normalization_1/gamma/Assign)^batch_normalization_1/moving_mean/Assign-^batch_normalization_1/moving_variance/Assign"^batch_normalization_2/beta/Assign#^batch_normalization_2/gamma/Assign)^batch_normalization_2/moving_mean/Assign-^batch_normalization_2/moving_variance/Assign"^batch_normalization_3/beta/Assign#^batch_normalization_3/gamma/Assign)^batch_normalization_3/moving_mean/Assign-^batch_normalization_3/moving_variance/Assign"^batch_normalization_4/beta/Assign#^batch_normalization_4/gamma/Assign)^batch_normalization_4/moving_mean/Assign-^batch_normalization_4/moving_variance/Assign^conv2d/bias/Assign^conv2d/kernel/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^conv2d_3/bias/Assign^conv2d_3/kernel/Assign^conv2d_4/bias/Assign^conv2d_4/kernel/Assign^dense/bias/Assign^dense/kernel/Assign^prediction/bias/Assign^prediction/kernel/Assign
Ж
prediction_targetPlaceholder*
dtype0*0
_output_shapes
:                  *%
shape:                  
v
total/Initializer/zerosConst*
valueB
 *    *
_class

loc:@total*
dtype0*
_output_shapes
: 
Й
totalVarHandleOp*
_output_shapes
: *
shared_nametotal*
_class

loc:@total*
	container *
shape: *
dtype0
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
g
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
_class

loc:@total*
dtype0
q
total/Read/ReadVariableOpReadVariableOptotal*
_class

loc:@total*
dtype0*
_output_shapes
: 
v
count/Initializer/zerosConst*
valueB
 *    *
_class

loc:@count*
dtype0*
_output_shapes
: 
Й
countVarHandleOp*
shared_namecount*
_class

loc:@count*
	container *
shape: *
dtype0*
_output_shapes
: 
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
g
count/AssignAssignVariableOpcountcount/Initializer/zeros*
_class

loc:@count*
dtype0
q
count/Read/ReadVariableOpReadVariableOpcount*
_class

loc:@count*
dtype0*
_output_shapes
: 
g
metrics/acc/ArgMax/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
Ъ
metrics/acc/ArgMaxArgMaxprediction_targetmetrics/acc/ArgMax/dimension*#
_output_shapes
:         *

Tidx0*
T0*
output_type0	
i
metrics/acc/ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
         
Я
metrics/acc/ArgMax_1ArgMaxprediction/Softmaxmetrics/acc/ArgMax_1/dimension*#
_output_shapes
:         *

Tidx0*
T0*
output_type0	
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*#
_output_shapes
:         *
T0	
x
metrics/acc/CastCastmetrics/acc/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:         *

DstT0
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
y
metrics/acc/SumSummetrics/acc/Castmetrics/acc/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0
М
metrics/acc/ReadVariableOpReadVariableOptotal ^metrics/acc/AssignAddVariableOp^metrics/acc/Sum*
dtype0*
_output_shapes
: 
[
metrics/acc/SizeSizemetrics/acc/Cast*
T0*
out_type0*
_output_shapes
: 
l
metrics/acc/Cast_1Castmetrics/acc/Size*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
В
!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_1 ^metrics/acc/AssignAddVariableOp*
dtype0
а
metrics/acc/ReadVariableOp_1ReadVariableOpcount ^metrics/acc/AssignAddVariableOp"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
З
%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
Й
'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1*
_output_shapes
: *
dtype0
У
metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
Y
metrics/acc/IdentityIdentitymetrics/acc/div_no_nan*
T0*
_output_shapes
: 
_
loss/prediction_loss/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
}
;loss/prediction_loss/softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0*
_output_shapes
: 
О
<loss/prediction_loss/softmax_cross_entropy_with_logits/ShapeShapeprediction/BiasAdd*
_output_shapes
:*
T0*
out_type0

=loss/prediction_loss/softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
Р
>loss/prediction_loss/softmax_cross_entropy_with_logits/Shape_1Shapeprediction/BiasAdd*
T0*
out_type0*
_output_shapes
:
~
<loss/prediction_loss/softmax_cross_entropy_with_logits/Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
▀
:loss/prediction_loss/softmax_cross_entropy_with_logits/SubSub=loss/prediction_loss/softmax_cross_entropy_with_logits/Rank_1<loss/prediction_loss/softmax_cross_entropy_with_logits/Sub/y*
_output_shapes
: *
T0
└
Bloss/prediction_loss/softmax_cross_entropy_with_logits/Slice/beginPack:loss/prediction_loss/softmax_cross_entropy_with_logits/Sub*
N*
_output_shapes
:*
T0*

axis 
Л
Aloss/prediction_loss/softmax_cross_entropy_with_logits/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
╛
<loss/prediction_loss/softmax_cross_entropy_with_logits/SliceSlice>loss/prediction_loss/softmax_cross_entropy_with_logits/Shape_1Bloss/prediction_loss/softmax_cross_entropy_with_logits/Slice/beginAloss/prediction_loss/softmax_cross_entropy_with_logits/Slice/size*
Index0*
T0*
_output_shapes
:
Щ
Floss/prediction_loss/softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
Д
Bloss/prediction_loss/softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
═
=loss/prediction_loss/softmax_cross_entropy_with_logits/concatConcatV2Floss/prediction_loss/softmax_cross_entropy_with_logits/concat/values_0<loss/prediction_loss/softmax_cross_entropy_with_logits/SliceBloss/prediction_loss/softmax_cross_entropy_with_logits/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
х
>loss/prediction_loss/softmax_cross_entropy_with_logits/ReshapeReshapeprediction/BiasAdd=loss/prediction_loss/softmax_cross_entropy_with_logits/concat*
T0*
Tshape0*0
_output_shapes
:                  

=loss/prediction_loss/softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
П
>loss/prediction_loss/softmax_cross_entropy_with_logits/Shape_2Shapeprediction_target*
T0*
out_type0*
_output_shapes
:
А
>loss/prediction_loss/softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
у
<loss/prediction_loss/softmax_cross_entropy_with_logits/Sub_1Sub=loss/prediction_loss/softmax_cross_entropy_with_logits/Rank_2>loss/prediction_loss/softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
─
Dloss/prediction_loss/softmax_cross_entropy_with_logits/Slice_1/beginPack<loss/prediction_loss/softmax_cross_entropy_with_logits/Sub_1*

axis *
N*
_output_shapes
:*
T0
Н
Closs/prediction_loss/softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
─
>loss/prediction_loss/softmax_cross_entropy_with_logits/Slice_1Slice>loss/prediction_loss/softmax_cross_entropy_with_logits/Shape_2Dloss/prediction_loss/softmax_cross_entropy_with_logits/Slice_1/beginCloss/prediction_loss/softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:
Ы
Hloss/prediction_loss/softmax_cross_entropy_with_logits/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
         
Ж
Dloss/prediction_loss/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╒
?loss/prediction_loss/softmax_cross_entropy_with_logits/concat_1ConcatV2Hloss/prediction_loss/softmax_cross_entropy_with_logits/concat_1/values_0>loss/prediction_loss/softmax_cross_entropy_with_logits/Slice_1Dloss/prediction_loss/softmax_cross_entropy_with_logits/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
ш
@loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_1Reshapeprediction_target?loss/prediction_loss/softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0*0
_output_shapes
:                  
г
6loss/prediction_loss/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits>loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape@loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_1*
T0*?
_output_shapes-
+:         :                  
А
>loss/prediction_loss/softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
с
<loss/prediction_loss/softmax_cross_entropy_with_logits/Sub_2Sub;loss/prediction_loss/softmax_cross_entropy_with_logits/Rank>loss/prediction_loss/softmax_cross_entropy_with_logits/Sub_2/y*
_output_shapes
: *
T0
О
Dloss/prediction_loss/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
├
Closs/prediction_loss/softmax_cross_entropy_with_logits/Slice_2/sizePack<loss/prediction_loss/softmax_cross_entropy_with_logits/Sub_2*
T0*

axis *
N*
_output_shapes
:
┬
>loss/prediction_loss/softmax_cross_entropy_with_logits/Slice_2Slice<loss/prediction_loss/softmax_cross_entropy_with_logits/ShapeDloss/prediction_loss/softmax_cross_entropy_with_logits/Slice_2/beginCloss/prediction_loss/softmax_cross_entropy_with_logits/Slice_2/size*
_output_shapes
:*
Index0*
T0
 
@loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_2Reshape6loss/prediction_loss/softmax_cross_entropy_with_logits>loss/prediction_loss/softmax_cross_entropy_with_logits/Slice_2*#
_output_shapes
:         *
T0*
Tshape0
m
(loss/prediction_loss/weighted_loss/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ъ
Wloss/prediction_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Ш
Vloss/prediction_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
_output_shapes
: *
value	B : *
dtype0
╓
Vloss/prediction_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape@loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_2*
_output_shapes
:*
T0*
out_type0
Ч
Uloss/prediction_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
m
eloss/prediction_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
м
Dloss/prediction_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape@loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_2f^loss/prediction_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
T0*
out_type0
ё
Dloss/prediction_loss/weighted_loss/broadcast_weights/ones_like/ConstConstf^loss/prediction_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Т
>loss/prediction_loss/weighted_loss/broadcast_weights/ones_likeFillDloss/prediction_loss/weighted_loss/broadcast_weights/ones_like/ShapeDloss/prediction_loss/weighted_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:         
╙
4loss/prediction_loss/weighted_loss/broadcast_weightsMul(loss/prediction_loss/weighted_loss/Const>loss/prediction_loss/weighted_loss/broadcast_weights/ones_like*#
_output_shapes
:         *
T0
╙
&loss/prediction_loss/weighted_loss/MulMul@loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_24loss/prediction_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:         
f
loss/prediction_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
г
loss/prediction_loss/SumSum&loss/prediction_loss/weighted_loss/Mulloss/prediction_loss/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
В
!loss/prediction_loss/num_elementsSize&loss/prediction_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
: 
С
&loss/prediction_loss/num_elements/CastCast!loss/prediction_loss/num_elements*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
_
loss/prediction_loss/Const_2Const*
valueB *
dtype0*
_output_shapes
: 
Ч
loss/prediction_loss/Sum_1Sumloss/prediction_loss/Sumloss/prediction_loss/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Л
loss/prediction_loss/valueDivNoNanloss/prediction_loss/Sum_1&loss/prediction_loss/num_elements/Cast*
_output_shapes
: *
T0
O

loss/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
X
loss/mulMul
loss/mul/xloss/prediction_loss/value*
T0*
_output_shapes
: 
i
$loss/conv2d/kernel/Regularizer/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
&loss/conv2d_1/kernel/Regularizer/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
&loss/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
k
&loss/conv2d_3/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
k
&loss/conv2d_4/kernel/Regularizer/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
&loss/conv2d/kernel/Regularizer_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
(loss/conv2d_1/kernel/Regularizer_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
m
(loss/conv2d_2/kernel/Regularizer_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
(loss/conv2d_3/kernel/Regularizer_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
(loss/conv2d_4/kernel/Regularizer_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Б
	loss/AddNAddN$loss/conv2d/kernel/Regularizer/Const&loss/conv2d_1/kernel/Regularizer/Const&loss/conv2d_2/kernel/Regularizer/Const&loss/conv2d_3/kernel/Regularizer/Const&loss/conv2d_4/kernel/Regularizer/Const*
T0*
N*
_output_shapes
: 
E
loss/addAddloss/mul	loss/AddN*
_output_shapes
: *
T0
V
SGD/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
\
SGD/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
{
SGD/gradients/FillFillSGD/gradients/ShapeSGD/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
w
SGD/gradients/loss/mul_grad/MulMulSGD/gradients/Fillloss/prediction_loss/value*
_output_shapes
: *
T0
i
!SGD/gradients/loss/mul_grad/Mul_1MulSGD/gradients/Fill
loss/mul/x*
T0*
_output_shapes
: 
v
3SGD/gradients/loss/prediction_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
x
5SGD/gradients/loss/prediction_loss/value_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
Е
CSGD/gradients/loss/prediction_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgs3SGD/gradients/loss/prediction_loss/value_grad/Shape5SGD/gradients/loss/prediction_loss/value_grad/Shape_1*
T0*2
_output_shapes 
:         :         
░
8SGD/gradients/loss/prediction_loss/value_grad/div_no_nanDivNoNan!SGD/gradients/loss/mul_grad/Mul_1&loss/prediction_loss/num_elements/Cast*
T0*
_output_shapes
: 
ї
1SGD/gradients/loss/prediction_loss/value_grad/SumSum8SGD/gradients/loss/prediction_loss/value_grad/div_no_nanCSGD/gradients/loss/prediction_loss/value_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
╫
5SGD/gradients/loss/prediction_loss/value_grad/ReshapeReshape1SGD/gradients/loss/prediction_loss/value_grad/Sum3SGD/gradients/loss/prediction_loss/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
u
1SGD/gradients/loss/prediction_loss/value_grad/NegNegloss/prediction_loss/Sum_1*
_output_shapes
: *
T0
┬
:SGD/gradients/loss/prediction_loss/value_grad/div_no_nan_1DivNoNan1SGD/gradients/loss/prediction_loss/value_grad/Neg&loss/prediction_loss/num_elements/Cast*
_output_shapes
: *
T0
╦
:SGD/gradients/loss/prediction_loss/value_grad/div_no_nan_2DivNoNan:SGD/gradients/loss/prediction_loss/value_grad/div_no_nan_1&loss/prediction_loss/num_elements/Cast*
T0*
_output_shapes
: 
╕
1SGD/gradients/loss/prediction_loss/value_grad/mulMul!SGD/gradients/loss/mul_grad/Mul_1:SGD/gradients/loss/prediction_loss/value_grad/div_no_nan_2*
_output_shapes
: *
T0
Є
3SGD/gradients/loss/prediction_loss/value_grad/Sum_1Sum1SGD/gradients/loss/prediction_loss/value_grad/mulESGD/gradients/loss/prediction_loss/value_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
▌
7SGD/gradients/loss/prediction_loss/value_grad/Reshape_1Reshape3SGD/gradients/loss/prediction_loss/value_grad/Sum_15SGD/gradients/loss/prediction_loss/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
~
;SGD/gradients/loss/prediction_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
у
5SGD/gradients/loss/prediction_loss/Sum_1_grad/ReshapeReshape5SGD/gradients/loss/prediction_loss/value_grad/Reshape;SGD/gradients/loss/prediction_loss/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
v
3SGD/gradients/loss/prediction_loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
┘
2SGD/gradients/loss/prediction_loss/Sum_1_grad/TileTile5SGD/gradients/loss/prediction_loss/Sum_1_grad/Reshape3SGD/gradients/loss/prediction_loss/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
Г
9SGD/gradients/loss/prediction_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
р
3SGD/gradients/loss/prediction_loss/Sum_grad/ReshapeReshape2SGD/gradients/loss/prediction_loss/Sum_1_grad/Tile9SGD/gradients/loss/prediction_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
Ч
1SGD/gradients/loss/prediction_loss/Sum_grad/ShapeShape&loss/prediction_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
:
р
0SGD/gradients/loss/prediction_loss/Sum_grad/TileTile3SGD/gradients/loss/prediction_loss/Sum_grad/Reshape1SGD/gradients/loss/prediction_loss/Sum_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
┐
?SGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/ShapeShape@loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_2*
T0*
out_type0*
_output_shapes
:
╡
ASGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/Shape_1Shape4loss/prediction_loss/weighted_loss/broadcast_weights*
T0*
out_type0*
_output_shapes
:
й
OSGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs?SGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/ShapeASGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┌
=SGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/MulMul0SGD/gradients/loss/prediction_loss/Sum_grad/Tile4loss/prediction_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:         
Ф
=SGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/SumSum=SGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/MulOSGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
И
ASGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/ReshapeReshape=SGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/Sum?SGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/Shape*#
_output_shapes
:         *
T0*
Tshape0
ш
?SGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/Mul_1Mul@loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_20SGD/gradients/loss/prediction_loss/Sum_grad/Tile*#
_output_shapes
:         *
T0
Ъ
?SGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/Sum_1Sum?SGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/Mul_1QSGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
О
CSGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/Reshape_1Reshape?SGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/Sum_1ASGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/Shape_1*#
_output_shapes
:         *
T0*
Tshape0
╧
YSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShape6loss/prediction_loss/softmax_cross_entropy_with_logits*
T0*
out_type0*
_output_shapes
:
└
[SGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapeASGD/gradients/loss/prediction_loss/weighted_loss/Mul_grad/ReshapeYSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
Ъ
SGD/gradients/zeros_like	ZerosLike8loss/prediction_loss/softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:                  
г
XSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
╫
TSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDims[SGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeXSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
Я
MSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/mulMulTSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/ExpandDims8loss/prediction_loss/softmax_cross_entropy_with_logits:1*0
_output_shapes
:                  *
T0
▌
TSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax>loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape*0
_output_shapes
:                  *
T0
х
MSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/NegNegTSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/LogSoftmax*0
_output_shapes
:                  *
T0
е
ZSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
█
VSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDims[SGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeZSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*'
_output_shapes
:         *

Tdim0*
T0
╕
OSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/mul_1MulVSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1MSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/Neg*
T0*0
_output_shapes
:                  
й
WSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapeprediction/BiasAdd*
_output_shapes
:*
T0*
out_type0
╠
YSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapeMSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits_grad/mulWSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
╫
1SGD/gradients/prediction/BiasAdd_grad/BiasAddGradBiasAddGradYSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
Л
+SGD/gradients/prediction/MatMul_grad/MatMulMatMulYSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshape prediction/MatMul/ReadVariableOp*(
_output_shapes
:         А*
transpose_a( *
transpose_b(*
T0
ё
-SGD/gradients/prediction/MatMul_grad/MatMul_1MatMuldense/SigmoidYSGD/gradients/loss/prediction_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
_output_shapes
:	А*
transpose_a(*
transpose_b( *
T0
y
SGD/iter/Initializer/zerosConst*
value	B	 R *
_class
loc:@SGD/iter*
dtype0	*
_output_shapes
: 
Т
SGD/iterVarHandleOp*
_class
loc:@SGD/iter*
	container *
shape: *
dtype0	*
_output_shapes
: *
shared_name
SGD/iter
a
)SGD/iter/IsInitialized/VarIsInitializedOpVarIsInitializedOpSGD/iter*
_output_shapes
: 
s
SGD/iter/AssignAssignVariableOpSGD/iterSGD/iter/Initializer/zeros*
_class
loc:@SGD/iter*
dtype0	
z
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
_class
loc:@SGD/iter*
dtype0	
Ж
#SGD/decay/Initializer/initial_valueConst*
valueB
 *    *
_class
loc:@SGD/decay*
dtype0*
_output_shapes
: 
Х
	SGD/decayVarHandleOp*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name	SGD/decay*
_class
loc:@SGD/decay
c
*SGD/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOp	SGD/decay*
_output_shapes
: 

SGD/decay/AssignAssignVariableOp	SGD/decay#SGD/decay/Initializer/initial_value*
_class
loc:@SGD/decay*
dtype0
}
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_class
loc:@SGD/decay*
dtype0*
_output_shapes
: 
Ц
+SGD/learning_rate/Initializer/initial_valueConst*
valueB
 *
╫#<*$
_class
loc:@SGD/learning_rate*
dtype0*
_output_shapes
: 
н
SGD/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *"
shared_nameSGD/learning_rate*$
_class
loc:@SGD/learning_rate*
	container *
shape: 
s
2SGD/learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOpSGD/learning_rate*
_output_shapes
: 
Я
SGD/learning_rate/AssignAssignVariableOpSGD/learning_rate+SGD/learning_rate/Initializer/initial_value*$
_class
loc:@SGD/learning_rate*
dtype0
Х
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
dtype0*
_output_shapes
: *$
_class
loc:@SGD/learning_rate
М
&SGD/momentum/Initializer/initial_valueConst*
valueB
 *fff?*
_class
loc:@SGD/momentum*
dtype0*
_output_shapes
: 
Ю
SGD/momentumVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameSGD/momentum*
_class
loc:@SGD/momentum*
	container *
shape: 
i
-SGD/momentum/IsInitialized/VarIsInitializedOpVarIsInitializedOpSGD/momentum*
_output_shapes
: 
Л
SGD/momentum/AssignAssignVariableOpSGD/momentum&SGD/momentum/Initializer/initial_value*
_class
loc:@SGD/momentum*
dtype0
Ж
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_class
loc:@SGD/momentum*
dtype0*
_output_shapes
: 
╖
@SGD/prediction/kernel/momentum/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@prediction/kernel*
valueB"      *
dtype0*
_output_shapes
:
б
6SGD/prediction/kernel/momentum/Initializer/zeros/ConstConst*$
_class
loc:@prediction/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
0SGD/prediction/kernel/momentum/Initializer/zerosFill@SGD/prediction/kernel/momentum/Initializer/zeros/shape_as_tensor6SGD/prediction/kernel/momentum/Initializer/zeros/Const*
T0*$
_class
loc:@prediction/kernel*

index_type0*
_output_shapes
:	А
╨
SGD/prediction/kernel/momentumVarHandleOp*/
shared_name SGD/prediction/kernel/momentum*$
_class
loc:@prediction/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
: 
│
?SGD/prediction/kernel/momentum/IsInitialized/VarIsInitializedOpVarIsInitializedOpSGD/prediction/kernel/momentum*$
_class
loc:@prediction/kernel*
_output_shapes
: 
╛
%SGD/prediction/kernel/momentum/AssignAssignVariableOpSGD/prediction/kernel/momentum0SGD/prediction/kernel/momentum/Initializer/zeros*$
_class
loc:@prediction/kernel*
dtype0
╕
2SGD/prediction/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/prediction/kernel/momentum*
dtype0*
_output_shapes
:	А*$
_class
loc:@prediction/kernel
Я
.SGD/prediction/bias/momentum/Initializer/zerosConst*
_output_shapes
:*"
_class
loc:@prediction/bias*
valueB*    *
dtype0
┼
SGD/prediction/bias/momentumVarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *-
shared_nameSGD/prediction/bias/momentum*"
_class
loc:@prediction/bias
н
=SGD/prediction/bias/momentum/IsInitialized/VarIsInitializedOpVarIsInitializedOpSGD/prediction/bias/momentum*"
_class
loc:@prediction/bias*
_output_shapes
: 
╢
#SGD/prediction/bias/momentum/AssignAssignVariableOpSGD/prediction/bias/momentum.SGD/prediction/bias/momentum/Initializer/zeros*"
_class
loc:@prediction/bias*
dtype0
н
0SGD/prediction/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/prediction/bias/momentum*"
_class
loc:@prediction/bias*
dtype0*
_output_shapes
:
Ф
JSGD/SGD/update_prediction/kernel/ResourceApplyKerasMomentum/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
С
LSGD/SGD/update_prediction/kernel/ResourceApplyKerasMomentum/ReadVariableOp_1ReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
Е
;SGD/SGD/update_prediction/kernel/ResourceApplyKerasMomentumResourceApplyKerasMomentumprediction/kernelSGD/prediction/kernel/momentumJSGD/SGD/update_prediction/kernel/ResourceApplyKerasMomentum/ReadVariableOp-SGD/gradients/prediction/MatMul_grad/MatMul_1LSGD/SGD/update_prediction/kernel/ResourceApplyKerasMomentum/ReadVariableOp_1*
use_locking(*
T0*
use_nesterov( 
Т
HSGD/SGD/update_prediction/bias/ResourceApplyKerasMomentum/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
П
JSGD/SGD/update_prediction/bias/ResourceApplyKerasMomentum/ReadVariableOp_1ReadVariableOpSGD/momentum*
dtype0*
_output_shapes
: 
 
9SGD/SGD/update_prediction/bias/ResourceApplyKerasMomentumResourceApplyKerasMomentumprediction/biasSGD/prediction/bias/momentumHSGD/SGD/update_prediction/bias/ResourceApplyKerasMomentum/ReadVariableOp1SGD/gradients/prediction/BiasAdd_grad/BiasAddGradJSGD/SGD/update_prediction/bias/ResourceApplyKerasMomentum/ReadVariableOp_1*
use_locking(*
T0*
use_nesterov( 
╔
SGD/SGD/ConstConst:^SGD/SGD/update_prediction/bias/ResourceApplyKerasMomentum<^SGD/SGD/update_prediction/kernel/ResourceApplyKerasMomentum*
value	B	 R*
dtype0	*
_output_shapes
: 
X
SGD/SGD/AssignAddVariableOpAssignAddVariableOpSGD/iterSGD/SGD/Const*
dtype0	
я
SGD/SGD/ReadVariableOpReadVariableOpSGD/iter^SGD/SGD/AssignAddVariableOp:^SGD/SGD/update_prediction/bias/ResourceApplyKerasMomentum<^SGD/SGD/update_prediction/kernel/ResourceApplyKerasMomentum*
dtype0	*
_output_shapes
: 
F
training_1/group_depsNoOp^SGD/SGD/AssignAddVariableOp	^loss/add
V
VarIsInitializedOp_34VarIsInitializedOpSGD/learning_rate*
_output_shapes
: 
c
VarIsInitializedOp_35VarIsInitializedOpSGD/prediction/kernel/momentum*
_output_shapes
: 
a
VarIsInitializedOp_36VarIsInitializedOpSGD/prediction/bias/momentum*
_output_shapes
: 
N
VarIsInitializedOp_37VarIsInitializedOp	SGD/decay*
_output_shapes
: 
M
VarIsInitializedOp_38VarIsInitializedOpSGD/iter*
_output_shapes
: 
J
VarIsInitializedOp_39VarIsInitializedOptotal*
_output_shapes
: 
Q
VarIsInitializedOp_40VarIsInitializedOpSGD/momentum*
_output_shapes
: 
J
VarIsInitializedOp_41VarIsInitializedOpcount*
_output_shapes
: 
╨
init_1NoOp^SGD/decay/Assign^SGD/iter/Assign^SGD/learning_rate/Assign^SGD/momentum/Assign$^SGD/prediction/bias/momentum/Assign&^SGD/prediction/kernel/momentum/Assign^count/Assign^total/Assign
O
Placeholder_34Placeholder*
dtype0	*
_output_shapes
: *
shape: 
N
AssignVariableOp_34AssignVariableOpSGD/iterPlaceholder_34*
dtype0	
h
ReadVariableOp_34ReadVariableOpSGD/iter^AssignVariableOp_34*
_output_shapes
: *
dtype0	
a
Placeholder_35Placeholder*
shape:	А*
dtype0*
_output_shapes
:	А
d
AssignVariableOp_35AssignVariableOpSGD/prediction/kernel/momentumPlaceholder_35*
dtype0
З
ReadVariableOp_35ReadVariableOpSGD/prediction/kernel/momentum^AssignVariableOp_35*
dtype0*
_output_shapes
:	А
W
Placeholder_36Placeholder*
dtype0*
_output_shapes
:*
shape:
b
AssignVariableOp_36AssignVariableOpSGD/prediction/bias/momentumPlaceholder_36*
dtype0
А
ReadVariableOp_36ReadVariableOpSGD/prediction/bias/momentum^AssignVariableOp_36*
dtype0*
_output_shapes
:
И
prediction_target_1Placeholder*
dtype0*0
_output_shapes
:                  *%
shape:                  
z
total_1/Initializer/zerosConst*
valueB
 *    *
_class
loc:@total_1*
dtype0*
_output_shapes
: 
П
total_1VarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_name	total_1*
_class
loc:@total_1*
	container 
_
(total_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal_1*
_output_shapes
: 
o
total_1/AssignAssignVariableOptotal_1total_1/Initializer/zeros*
_class
loc:@total_1*
dtype0
w
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_class
loc:@total_1*
dtype0*
_output_shapes
: 
z
count_1/Initializer/zerosConst*
valueB
 *    *
_class
loc:@count_1*
dtype0*
_output_shapes
: 
П
count_1VarHandleOp*
dtype0*
_output_shapes
: *
shared_name	count_1*
_class
loc:@count_1*
	container *
shape: 
_
(count_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount_1*
_output_shapes
: 
o
count_1/AssignAssignVariableOpcount_1count_1/Initializer/zeros*
_class
loc:@count_1*
dtype0
w
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_class
loc:@count_1*
dtype0*
_output_shapes
: 
i
metrics_2/acc/ArgMax/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
а
metrics_2/acc/ArgMaxArgMaxprediction_target_1metrics_2/acc/ArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:         
k
 metrics_2/acc/ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
         
г
metrics_2/acc/ArgMax_1ArgMaxprediction/Softmax metrics_2/acc/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:         *

Tidx0
x
metrics_2/acc/EqualEqualmetrics_2/acc/ArgMaxmetrics_2/acc/ArgMax_1*
T0	*#
_output_shapes
:         
|
metrics_2/acc/CastCastmetrics_2/acc/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:         *

DstT0
]
metrics_2/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:

metrics_2/acc/SumSummetrics_2/acc/Castmetrics_2/acc/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
a
!metrics_2/acc/AssignAddVariableOpAssignAddVariableOptotal_1metrics_2/acc/Sum*
dtype0
Ф
metrics_2/acc/ReadVariableOpReadVariableOptotal_1"^metrics_2/acc/AssignAddVariableOp^metrics_2/acc/Sum*
dtype0*
_output_shapes
: 
_
metrics_2/acc/SizeSizemetrics_2/acc/Cast*
T0*
out_type0*
_output_shapes
: 
p
metrics_2/acc/Cast_1Castmetrics_2/acc/Size*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
К
#metrics_2/acc/AssignAddVariableOp_1AssignAddVariableOpcount_1metrics_2/acc/Cast_1"^metrics_2/acc/AssignAddVariableOp*
dtype0
и
metrics_2/acc/ReadVariableOp_1ReadVariableOpcount_1"^metrics_2/acc/AssignAddVariableOp$^metrics_2/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
Н
'metrics_2/acc/div_no_nan/ReadVariableOpReadVariableOptotal_1$^metrics_2/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
П
)metrics_2/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount_1$^metrics_2/acc/AssignAddVariableOp_1*
_output_shapes
: *
dtype0
Щ
metrics_2/acc/div_no_nanDivNoNan'metrics_2/acc/div_no_nan/ReadVariableOp)metrics_2/acc/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
]
metrics_2/acc/IdentityIdentitymetrics_2/acc/div_no_nan*
T0*
_output_shapes
: 
a
loss_1/prediction_loss/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

=loss_1/prediction_loss/softmax_cross_entropy_with_logits/RankConst*
dtype0*
_output_shapes
: *
value	B :
Р
>loss_1/prediction_loss/softmax_cross_entropy_with_logits/ShapeShapeprediction/BiasAdd*
T0*
out_type0*
_output_shapes
:
Б
?loss_1/prediction_loss/softmax_cross_entropy_with_logits/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
Т
@loss_1/prediction_loss/softmax_cross_entropy_with_logits/Shape_1Shapeprediction/BiasAdd*
T0*
out_type0*
_output_shapes
:
А
>loss_1/prediction_loss/softmax_cross_entropy_with_logits/Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
х
<loss_1/prediction_loss/softmax_cross_entropy_with_logits/SubSub?loss_1/prediction_loss/softmax_cross_entropy_with_logits/Rank_1>loss_1/prediction_loss/softmax_cross_entropy_with_logits/Sub/y*
_output_shapes
: *
T0
─
Dloss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice/beginPack<loss_1/prediction_loss/softmax_cross_entropy_with_logits/Sub*
T0*

axis *
N*
_output_shapes
:
Н
Closs_1/prediction_loss/softmax_cross_entropy_with_logits/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
╞
>loss_1/prediction_loss/softmax_cross_entropy_with_logits/SliceSlice@loss_1/prediction_loss/softmax_cross_entropy_with_logits/Shape_1Dloss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice/beginCloss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice/size*
Index0*
T0*
_output_shapes
:
Ы
Hloss_1/prediction_loss/softmax_cross_entropy_with_logits/concat/values_0Const*
_output_shapes
:*
valueB:
         *
dtype0
Ж
Dloss_1/prediction_loss/softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╒
?loss_1/prediction_loss/softmax_cross_entropy_with_logits/concatConcatV2Hloss_1/prediction_loss/softmax_cross_entropy_with_logits/concat/values_0>loss_1/prediction_loss/softmax_cross_entropy_with_logits/SliceDloss_1/prediction_loss/softmax_cross_entropy_with_logits/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
щ
@loss_1/prediction_loss/softmax_cross_entropy_with_logits/ReshapeReshapeprediction/BiasAdd?loss_1/prediction_loss/softmax_cross_entropy_with_logits/concat*
T0*
Tshape0*0
_output_shapes
:                  
Б
?loss_1/prediction_loss/softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
У
@loss_1/prediction_loss/softmax_cross_entropy_with_logits/Shape_2Shapeprediction_target_1*
T0*
out_type0*
_output_shapes
:
В
@loss_1/prediction_loss/softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
щ
>loss_1/prediction_loss/softmax_cross_entropy_with_logits/Sub_1Sub?loss_1/prediction_loss/softmax_cross_entropy_with_logits/Rank_2@loss_1/prediction_loss/softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
╚
Floss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice_1/beginPack>loss_1/prediction_loss/softmax_cross_entropy_with_logits/Sub_1*
T0*

axis *
N*
_output_shapes
:
П
Eloss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
╠
@loss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice_1Slice@loss_1/prediction_loss/softmax_cross_entropy_with_logits/Shape_2Floss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice_1/beginEloss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:
Э
Jloss_1/prediction_loss/softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
И
Floss_1/prediction_loss/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
▌
Aloss_1/prediction_loss/softmax_cross_entropy_with_logits/concat_1ConcatV2Jloss_1/prediction_loss/softmax_cross_entropy_with_logits/concat_1/values_0@loss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice_1Floss_1/prediction_loss/softmax_cross_entropy_with_logits/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
ю
Bloss_1/prediction_loss/softmax_cross_entropy_with_logits/Reshape_1Reshapeprediction_target_1Aloss_1/prediction_loss/softmax_cross_entropy_with_logits/concat_1*0
_output_shapes
:                  *
T0*
Tshape0
й
8loss_1/prediction_loss/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits@loss_1/prediction_loss/softmax_cross_entropy_with_logits/ReshapeBloss_1/prediction_loss/softmax_cross_entropy_with_logits/Reshape_1*
T0*?
_output_shapes-
+:         :                  
В
@loss_1/prediction_loss/softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
ч
>loss_1/prediction_loss/softmax_cross_entropy_with_logits/Sub_2Sub=loss_1/prediction_loss/softmax_cross_entropy_with_logits/Rank@loss_1/prediction_loss/softmax_cross_entropy_with_logits/Sub_2/y*
_output_shapes
: *
T0
Р
Floss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
╟
Eloss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice_2/sizePack>loss_1/prediction_loss/softmax_cross_entropy_with_logits/Sub_2*
T0*

axis *
N*
_output_shapes
:
╩
@loss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice_2Slice>loss_1/prediction_loss/softmax_cross_entropy_with_logits/ShapeFloss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice_2/beginEloss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice_2/size*
Index0*
T0*
_output_shapes
:
Е
Bloss_1/prediction_loss/softmax_cross_entropy_with_logits/Reshape_2Reshape8loss_1/prediction_loss/softmax_cross_entropy_with_logits@loss_1/prediction_loss/softmax_cross_entropy_with_logits/Slice_2*#
_output_shapes
:         *
T0*
Tshape0
o
*loss_1/prediction_loss/weighted_loss/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ь
Yloss_1/prediction_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Ъ
Xloss_1/prediction_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
┌
Xloss_1/prediction_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShapeBloss_1/prediction_loss/softmax_cross_entropy_with_logits/Reshape_2*
T0*
out_type0*
_output_shapes
:
Щ
Wloss_1/prediction_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
o
gloss_1/prediction_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
▓
Floss_1/prediction_loss/weighted_loss/broadcast_weights/ones_like/ShapeShapeBloss_1/prediction_loss/softmax_cross_entropy_with_logits/Reshape_2h^loss_1/prediction_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
ї
Floss_1/prediction_loss/weighted_loss/broadcast_weights/ones_like/ConstConsth^loss_1/prediction_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ш
@loss_1/prediction_loss/weighted_loss/broadcast_weights/ones_likeFillFloss_1/prediction_loss/weighted_loss/broadcast_weights/ones_like/ShapeFloss_1/prediction_loss/weighted_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:         
┘
6loss_1/prediction_loss/weighted_loss/broadcast_weightsMul*loss_1/prediction_loss/weighted_loss/Const@loss_1/prediction_loss/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:         
┘
(loss_1/prediction_loss/weighted_loss/MulMulBloss_1/prediction_loss/softmax_cross_entropy_with_logits/Reshape_26loss_1/prediction_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:         
h
loss_1/prediction_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
й
loss_1/prediction_loss/SumSum(loss_1/prediction_loss/weighted_loss/Mulloss_1/prediction_loss/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Ж
#loss_1/prediction_loss/num_elementsSize(loss_1/prediction_loss/weighted_loss/Mul*
_output_shapes
: *
T0*
out_type0
Х
(loss_1/prediction_loss/num_elements/CastCast#loss_1/prediction_loss/num_elements*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
a
loss_1/prediction_loss/Const_2Const*
_output_shapes
: *
valueB *
dtype0
Э
loss_1/prediction_loss/Sum_1Sumloss_1/prediction_loss/Sumloss_1/prediction_loss/Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
С
loss_1/prediction_loss/valueDivNoNanloss_1/prediction_loss/Sum_1(loss_1/prediction_loss/num_elements/Cast*
_output_shapes
: *
T0
Q
loss_1/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
^

loss_1/mulMulloss_1/mul/xloss_1/prediction_loss/value*
_output_shapes
: *
T0
k
&loss_1/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
m
(loss_1/conv2d_1/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
m
(loss_1/conv2d_2/kernel/Regularizer/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
(loss_1/conv2d_3/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
m
(loss_1/conv2d_4/kernel/Regularizer/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
(loss_1/conv2d/kernel/Regularizer_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
*loss_1/conv2d_1/kernel/Regularizer_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
*loss_1/conv2d_2/kernel/Regularizer_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
*loss_1/conv2d_3/kernel/Regularizer_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
*loss_1/conv2d_4/kernel/Regularizer_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Н
loss_1/AddNAddN&loss_1/conv2d/kernel/Regularizer/Const(loss_1/conv2d_1/kernel/Regularizer/Const(loss_1/conv2d_2/kernel/Regularizer/Const(loss_1/conv2d_3/kernel/Regularizer/Const(loss_1/conv2d_4/kernel/Regularizer/Const*
N*
_output_shapes
: *
T0
K

loss_1/addAdd
loss_1/mulloss_1/AddN*
_output_shapes
: *
T0
L
VarIsInitializedOp_42VarIsInitializedOpcount_1*
_output_shapes
: 
L
VarIsInitializedOp_43VarIsInitializedOptotal_1*
_output_shapes
: 
0
init_2NoOp^count_1/Assign^total_1/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_d28dd17a17a6447d80adb0e7b4e16839/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
╖
save/SaveV2/tensor_namesConst*ъ
valueрB▌(B	SGD/decayBSGD/iterBSGD/learning_rateBSGD/momentumBSGD/prediction/bias/momentumBSGD/prediction/kernel/momentumBbatch_normalization/betaBbatch_normalization/gammaBbatch_normalization/moving_meanB#batch_normalization/moving_varianceBbatch_normalization_1/betaBbatch_normalization_1/gammaB!batch_normalization_1/moving_meanB%batch_normalization_1/moving_varianceBbatch_normalization_2/betaBbatch_normalization_2/gammaB!batch_normalization_2/moving_meanB%batch_normalization_2/moving_varianceBbatch_normalization_3/betaBbatch_normalization_3/gammaB!batch_normalization_3/moving_meanB%batch_normalization_3/moving_varianceBbatch_normalization_4/betaBbatch_normalization_4/gammaB!batch_normalization_4/moving_meanB%batch_normalization_4/moving_varianceBconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelBconv2d_2/biasBconv2d_2/kernelBconv2d_3/biasBconv2d_3/kernelBconv2d_4/biasBconv2d_4/kernelB
dense/biasBdense/kernelBprediction/biasBprediction/kernel*
dtype0*
_output_shapes
:(
│
save/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
Р
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesSGD/decay/Read/ReadVariableOpSGD/iter/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOp0SGD/prediction/bias/momentum/Read/ReadVariableOp2SGD/prediction/kernel/momentum/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp#prediction/bias/Read/ReadVariableOp%prediction/kernel/Read/ReadVariableOp*6
dtypes,
*2(	
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
_output_shapes
:*
T0*

axis 
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
║
save/RestoreV2/tensor_namesConst*ъ
valueрB▌(B	SGD/decayBSGD/iterBSGD/learning_rateBSGD/momentumBSGD/prediction/bias/momentumBSGD/prediction/kernel/momentumBbatch_normalization/betaBbatch_normalization/gammaBbatch_normalization/moving_meanB#batch_normalization/moving_varianceBbatch_normalization_1/betaBbatch_normalization_1/gammaB!batch_normalization_1/moving_meanB%batch_normalization_1/moving_varianceBbatch_normalization_2/betaBbatch_normalization_2/gammaB!batch_normalization_2/moving_meanB%batch_normalization_2/moving_varianceBbatch_normalization_3/betaBbatch_normalization_3/gammaB!batch_normalization_3/moving_meanB%batch_normalization_3/moving_varianceBbatch_normalization_4/betaBbatch_normalization_4/gammaB!batch_normalization_4/moving_meanB%batch_normalization_4/moving_varianceBconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelBconv2d_2/biasBconv2d_2/kernelBconv2d_3/biasBconv2d_3/kernelBconv2d_4/biasBconv2d_4/kernelB
dense/biasBdense/kernelBprediction/biasBprediction/kernel*
dtype0*
_output_shapes
:(
╢
save/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
╓
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*6
dtypes,
*2(	*╢
_output_shapesг
а::::::::::::::::::::::::::::::::::::::::
N
save/Identity_1Identitysave/RestoreV2*
_output_shapes
:*
T0
R
save/AssignVariableOpAssignVariableOp	SGD/decaysave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0	*
_output_shapes
:
S
save/AssignVariableOp_1AssignVariableOpSGD/itersave/Identity_2*
dtype0	
P
save/Identity_3Identitysave/RestoreV2:2*
_output_shapes
:*
T0
\
save/AssignVariableOp_2AssignVariableOpSGD/learning_ratesave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
_output_shapes
:*
T0
W
save/AssignVariableOp_3AssignVariableOpSGD/momentumsave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
g
save/AssignVariableOp_4AssignVariableOpSGD/prediction/bias/momentumsave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
i
save/AssignVariableOp_5AssignVariableOpSGD/prediction/kernel/momentumsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
_output_shapes
:*
T0
c
save/AssignVariableOp_6AssignVariableOpbatch_normalization/betasave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
d
save/AssignVariableOp_7AssignVariableOpbatch_normalization/gammasave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
_output_shapes
:*
T0
j
save/AssignVariableOp_8AssignVariableOpbatch_normalization/moving_meansave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
_output_shapes
:*
T0
o
save/AssignVariableOp_9AssignVariableOp#batch_normalization/moving_variancesave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
g
save/AssignVariableOp_10AssignVariableOpbatch_normalization_1/betasave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
h
save/AssignVariableOp_11AssignVariableOpbatch_normalization_1/gammasave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
_output_shapes
:*
T0
n
save/AssignVariableOp_12AssignVariableOp!batch_normalization_1/moving_meansave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
r
save/AssignVariableOp_13AssignVariableOp%batch_normalization_1/moving_variancesave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
T0*
_output_shapes
:
g
save/AssignVariableOp_14AssignVariableOpbatch_normalization_2/betasave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
h
save/AssignVariableOp_15AssignVariableOpbatch_normalization_2/gammasave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
_output_shapes
:*
T0
n
save/AssignVariableOp_16AssignVariableOp!batch_normalization_2/moving_meansave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
T0*
_output_shapes
:
r
save/AssignVariableOp_17AssignVariableOp%batch_normalization_2/moving_variancesave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
g
save/AssignVariableOp_18AssignVariableOpbatch_normalization_3/betasave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
_output_shapes
:*
T0
h
save/AssignVariableOp_19AssignVariableOpbatch_normalization_3/gammasave/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
T0*
_output_shapes
:
n
save/AssignVariableOp_20AssignVariableOp!batch_normalization_3/moving_meansave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
T0*
_output_shapes
:
r
save/AssignVariableOp_21AssignVariableOp%batch_normalization_3/moving_variancesave/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:
g
save/AssignVariableOp_22AssignVariableOpbatch_normalization_4/betasave/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
T0*
_output_shapes
:
h
save/AssignVariableOp_23AssignVariableOpbatch_normalization_4/gammasave/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
T0*
_output_shapes
:
n
save/AssignVariableOp_24AssignVariableOp!batch_normalization_4/moving_meansave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
_output_shapes
:*
T0
r
save/AssignVariableOp_25AssignVariableOp%batch_normalization_4/moving_variancesave/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:26*
T0*
_output_shapes
:
X
save/AssignVariableOp_26AssignVariableOpconv2d/biassave/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:27*
T0*
_output_shapes
:
Z
save/AssignVariableOp_27AssignVariableOpconv2d/kernelsave/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:28*
_output_shapes
:*
T0
Z
save/AssignVariableOp_28AssignVariableOpconv2d_1/biassave/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:29*
_output_shapes
:*
T0
\
save/AssignVariableOp_29AssignVariableOpconv2d_1/kernelsave/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:30*
T0*
_output_shapes
:
Z
save/AssignVariableOp_30AssignVariableOpconv2d_2/biassave/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:31*
T0*
_output_shapes
:
\
save/AssignVariableOp_31AssignVariableOpconv2d_2/kernelsave/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:32*
T0*
_output_shapes
:
Z
save/AssignVariableOp_32AssignVariableOpconv2d_3/biassave/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:33*
T0*
_output_shapes
:
\
save/AssignVariableOp_33AssignVariableOpconv2d_3/kernelsave/Identity_34*
dtype0
R
save/Identity_35Identitysave/RestoreV2:34*
_output_shapes
:*
T0
Z
save/AssignVariableOp_34AssignVariableOpconv2d_4/biassave/Identity_35*
dtype0
R
save/Identity_36Identitysave/RestoreV2:35*
_output_shapes
:*
T0
\
save/AssignVariableOp_35AssignVariableOpconv2d_4/kernelsave/Identity_36*
dtype0
R
save/Identity_37Identitysave/RestoreV2:36*
_output_shapes
:*
T0
W
save/AssignVariableOp_36AssignVariableOp
dense/biassave/Identity_37*
dtype0
R
save/Identity_38Identitysave/RestoreV2:37*
_output_shapes
:*
T0
Y
save/AssignVariableOp_37AssignVariableOpdense/kernelsave/Identity_38*
dtype0
R
save/Identity_39Identitysave/RestoreV2:38*
_output_shapes
:*
T0
\
save/AssignVariableOp_38AssignVariableOpprediction/biassave/Identity_39*
dtype0
R
save/Identity_40Identitysave/RestoreV2:39*
T0*
_output_shapes
:
^
save/AssignVariableOp_39AssignVariableOpprediction/kernelsave/Identity_40*
dtype0
╞
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_39^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard "&<
save/Const:0save/Identity:0save/restore_all (5 @F8"ВЗ
cond_contextЁЖьЖ
╚
"batch_normalization/cond/cond_text"batch_normalization/cond/pred_id:0#batch_normalization/cond/switch_t:0 *╓
activation/Relu:0
batch_normalization/beta:0
 batch_normalization/cond/Const:0
"batch_normalization/cond/Const_1:0
0batch_normalization/cond/FusedBatchNorm/Switch:1
)batch_normalization/cond/FusedBatchNorm:0
)batch_normalization/cond/FusedBatchNorm:1
)batch_normalization/cond/FusedBatchNorm:2
)batch_normalization/cond/FusedBatchNorm:3
)batch_normalization/cond/FusedBatchNorm:4
0batch_normalization/cond/ReadVariableOp/Switch:1
)batch_normalization/cond/ReadVariableOp:0
2batch_normalization/cond/ReadVariableOp_1/Switch:1
+batch_normalization/cond/ReadVariableOp_1:0
"batch_normalization/cond/pred_id:0
#batch_normalization/cond/switch_t:0
batch_normalization/gamma:0E
activation/Relu:00batch_normalization/cond/FusedBatchNorm/Switch:1P
batch_normalization/beta:02batch_normalization/cond/ReadVariableOp_1/Switch:1H
"batch_normalization/cond/pred_id:0"batch_normalization/cond/pred_id:0O
batch_normalization/gamma:00batch_normalization/cond/ReadVariableOp/Switch:1
╕
$batch_normalization/cond/cond_text_1"batch_normalization/cond/pred_id:0#batch_normalization/cond/switch_f:0*╞
activation/Relu:0
batch_normalization/beta:0
Abatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
:batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp:0
Cbatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
<batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1:0
2batch_normalization/cond/FusedBatchNorm_1/Switch:0
+batch_normalization/cond/FusedBatchNorm_1:0
+batch_normalization/cond/FusedBatchNorm_1:1
+batch_normalization/cond/FusedBatchNorm_1:2
+batch_normalization/cond/FusedBatchNorm_1:3
+batch_normalization/cond/FusedBatchNorm_1:4
2batch_normalization/cond/ReadVariableOp_2/Switch:0
+batch_normalization/cond/ReadVariableOp_2:0
2batch_normalization/cond/ReadVariableOp_3/Switch:0
+batch_normalization/cond/ReadVariableOp_3:0
"batch_normalization/cond/pred_id:0
#batch_normalization/cond/switch_f:0
batch_normalization/gamma:0
!batch_normalization/moving_mean:0
%batch_normalization/moving_variance:0G
activation/Relu:02batch_normalization/cond/FusedBatchNorm_1/Switch:0H
"batch_normalization/cond/pred_id:0"batch_normalization/cond/pred_id:0f
!batch_normalization/moving_mean:0Abatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0l
%batch_normalization/moving_variance:0Cbatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0P
batch_normalization/beta:02batch_normalization/cond/ReadVariableOp_3/Switch:0Q
batch_normalization/gamma:02batch_normalization/cond/ReadVariableOp_2/Switch:0
╖
$batch_normalization/cond_1/cond_text$batch_normalization/cond_1/pred_id:0%batch_normalization/cond_1/switch_t:0 *┐
"batch_normalization/cond_1/Const:0
$batch_normalization/cond_1/pred_id:0
%batch_normalization/cond_1/switch_t:0L
$batch_normalization/cond_1/pred_id:0$batch_normalization/cond_1/pred_id:0
╣
&batch_normalization/cond_1/cond_text_1$batch_normalization/cond_1/pred_id:0%batch_normalization/cond_1/switch_f:0*┴
$batch_normalization/cond_1/Const_1:0
$batch_normalization/cond_1/pred_id:0
%batch_normalization/cond_1/switch_f:0L
$batch_normalization/cond_1/pred_id:0$batch_normalization/cond_1/pred_id:0
А	
$batch_normalization_1/cond/cond_text$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_t:0 *И
activation_1/Relu:0
batch_normalization_1/beta:0
"batch_normalization_1/cond/Const:0
$batch_normalization_1/cond/Const_1:0
2batch_normalization_1/cond/FusedBatchNorm/Switch:1
+batch_normalization_1/cond/FusedBatchNorm:0
+batch_normalization_1/cond/FusedBatchNorm:1
+batch_normalization_1/cond/FusedBatchNorm:2
+batch_normalization_1/cond/FusedBatchNorm:3
+batch_normalization_1/cond/FusedBatchNorm:4
2batch_normalization_1/cond/ReadVariableOp/Switch:1
+batch_normalization_1/cond/ReadVariableOp:0
4batch_normalization_1/cond/ReadVariableOp_1/Switch:1
-batch_normalization_1/cond/ReadVariableOp_1:0
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_t:0
batch_normalization_1/gamma:0S
batch_normalization_1/gamma:02batch_normalization_1/cond/ReadVariableOp/Switch:1I
activation_1/Relu:02batch_normalization_1/cond/FusedBatchNorm/Switch:1T
batch_normalization_1/beta:04batch_normalization_1/cond/ReadVariableOp_1/Switch:1L
$batch_normalization_1/cond/pred_id:0$batch_normalization_1/cond/pred_id:0
А
&batch_normalization_1/cond/cond_text_1$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_f:0*И
activation_1/Relu:0
batch_normalization_1/beta:0
Cbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_1/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_1/cond/FusedBatchNorm_1:0
-batch_normalization_1/cond/FusedBatchNorm_1:1
-batch_normalization_1/cond/FusedBatchNorm_1:2
-batch_normalization_1/cond/FusedBatchNorm_1:3
-batch_normalization_1/cond/FusedBatchNorm_1:4
4batch_normalization_1/cond/ReadVariableOp_2/Switch:0
-batch_normalization_1/cond/ReadVariableOp_2:0
4batch_normalization_1/cond/ReadVariableOp_3/Switch:0
-batch_normalization_1/cond/ReadVariableOp_3:0
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_f:0
batch_normalization_1/gamma:0
#batch_normalization_1/moving_mean:0
'batch_normalization_1/moving_variance:0U
batch_normalization_1/gamma:04batch_normalization_1/cond/ReadVariableOp_2/Switch:0K
activation_1/Relu:04batch_normalization_1/cond/FusedBatchNorm_1/Switch:0L
$batch_normalization_1/cond/pred_id:0$batch_normalization_1/cond/pred_id:0p
'batch_normalization_1/moving_variance:0Ebatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0j
#batch_normalization_1/moving_mean:0Cbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0T
batch_normalization_1/beta:04batch_normalization_1/cond/ReadVariableOp_3/Switch:0
╟
&batch_normalization_1/cond_1/cond_text&batch_normalization_1/cond_1/pred_id:0'batch_normalization_1/cond_1/switch_t:0 *╔
$batch_normalization_1/cond_1/Const:0
&batch_normalization_1/cond_1/pred_id:0
'batch_normalization_1/cond_1/switch_t:0P
&batch_normalization_1/cond_1/pred_id:0&batch_normalization_1/cond_1/pred_id:0
╔
(batch_normalization_1/cond_1/cond_text_1&batch_normalization_1/cond_1/pred_id:0'batch_normalization_1/cond_1/switch_f:0*╦
&batch_normalization_1/cond_1/Const_1:0
&batch_normalization_1/cond_1/pred_id:0
'batch_normalization_1/cond_1/switch_f:0P
&batch_normalization_1/cond_1/pred_id:0&batch_normalization_1/cond_1/pred_id:0
А	
$batch_normalization_2/cond/cond_text$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_t:0 *И
activation_2/Relu:0
batch_normalization_2/beta:0
"batch_normalization_2/cond/Const:0
$batch_normalization_2/cond/Const_1:0
2batch_normalization_2/cond/FusedBatchNorm/Switch:1
+batch_normalization_2/cond/FusedBatchNorm:0
+batch_normalization_2/cond/FusedBatchNorm:1
+batch_normalization_2/cond/FusedBatchNorm:2
+batch_normalization_2/cond/FusedBatchNorm:3
+batch_normalization_2/cond/FusedBatchNorm:4
2batch_normalization_2/cond/ReadVariableOp/Switch:1
+batch_normalization_2/cond/ReadVariableOp:0
4batch_normalization_2/cond/ReadVariableOp_1/Switch:1
-batch_normalization_2/cond/ReadVariableOp_1:0
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_t:0
batch_normalization_2/gamma:0I
activation_2/Relu:02batch_normalization_2/cond/FusedBatchNorm/Switch:1S
batch_normalization_2/gamma:02batch_normalization_2/cond/ReadVariableOp/Switch:1T
batch_normalization_2/beta:04batch_normalization_2/cond/ReadVariableOp_1/Switch:1L
$batch_normalization_2/cond/pred_id:0$batch_normalization_2/cond/pred_id:0
А
&batch_normalization_2/cond/cond_text_1$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_f:0*И
activation_2/Relu:0
batch_normalization_2/beta:0
Cbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_2/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_2/cond/FusedBatchNorm_1:0
-batch_normalization_2/cond/FusedBatchNorm_1:1
-batch_normalization_2/cond/FusedBatchNorm_1:2
-batch_normalization_2/cond/FusedBatchNorm_1:3
-batch_normalization_2/cond/FusedBatchNorm_1:4
4batch_normalization_2/cond/ReadVariableOp_2/Switch:0
-batch_normalization_2/cond/ReadVariableOp_2:0
4batch_normalization_2/cond/ReadVariableOp_3/Switch:0
-batch_normalization_2/cond/ReadVariableOp_3:0
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_f:0
batch_normalization_2/gamma:0
#batch_normalization_2/moving_mean:0
'batch_normalization_2/moving_variance:0L
$batch_normalization_2/cond/pred_id:0$batch_normalization_2/cond/pred_id:0T
batch_normalization_2/beta:04batch_normalization_2/cond/ReadVariableOp_3/Switch:0K
activation_2/Relu:04batch_normalization_2/cond/FusedBatchNorm_1/Switch:0U
batch_normalization_2/gamma:04batch_normalization_2/cond/ReadVariableOp_2/Switch:0p
'batch_normalization_2/moving_variance:0Ebatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0j
#batch_normalization_2/moving_mean:0Cbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
╟
&batch_normalization_2/cond_1/cond_text&batch_normalization_2/cond_1/pred_id:0'batch_normalization_2/cond_1/switch_t:0 *╔
$batch_normalization_2/cond_1/Const:0
&batch_normalization_2/cond_1/pred_id:0
'batch_normalization_2/cond_1/switch_t:0P
&batch_normalization_2/cond_1/pred_id:0&batch_normalization_2/cond_1/pred_id:0
╔
(batch_normalization_2/cond_1/cond_text_1&batch_normalization_2/cond_1/pred_id:0'batch_normalization_2/cond_1/switch_f:0*╦
&batch_normalization_2/cond_1/Const_1:0
&batch_normalization_2/cond_1/pred_id:0
'batch_normalization_2/cond_1/switch_f:0P
&batch_normalization_2/cond_1/pred_id:0&batch_normalization_2/cond_1/pred_id:0
А	
$batch_normalization_3/cond/cond_text$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_t:0 *И
activation_3/Relu:0
batch_normalization_3/beta:0
"batch_normalization_3/cond/Const:0
$batch_normalization_3/cond/Const_1:0
2batch_normalization_3/cond/FusedBatchNorm/Switch:1
+batch_normalization_3/cond/FusedBatchNorm:0
+batch_normalization_3/cond/FusedBatchNorm:1
+batch_normalization_3/cond/FusedBatchNorm:2
+batch_normalization_3/cond/FusedBatchNorm:3
+batch_normalization_3/cond/FusedBatchNorm:4
2batch_normalization_3/cond/ReadVariableOp/Switch:1
+batch_normalization_3/cond/ReadVariableOp:0
4batch_normalization_3/cond/ReadVariableOp_1/Switch:1
-batch_normalization_3/cond/ReadVariableOp_1:0
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_t:0
batch_normalization_3/gamma:0I
activation_3/Relu:02batch_normalization_3/cond/FusedBatchNorm/Switch:1T
batch_normalization_3/beta:04batch_normalization_3/cond/ReadVariableOp_1/Switch:1L
$batch_normalization_3/cond/pred_id:0$batch_normalization_3/cond/pred_id:0S
batch_normalization_3/gamma:02batch_normalization_3/cond/ReadVariableOp/Switch:1
А
&batch_normalization_3/cond/cond_text_1$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_f:0*И
activation_3/Relu:0
batch_normalization_3/beta:0
Cbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_3/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_3/cond/FusedBatchNorm_1:0
-batch_normalization_3/cond/FusedBatchNorm_1:1
-batch_normalization_3/cond/FusedBatchNorm_1:2
-batch_normalization_3/cond/FusedBatchNorm_1:3
-batch_normalization_3/cond/FusedBatchNorm_1:4
4batch_normalization_3/cond/ReadVariableOp_2/Switch:0
-batch_normalization_3/cond/ReadVariableOp_2:0
4batch_normalization_3/cond/ReadVariableOp_3/Switch:0
-batch_normalization_3/cond/ReadVariableOp_3:0
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_f:0
batch_normalization_3/gamma:0
#batch_normalization_3/moving_mean:0
'batch_normalization_3/moving_variance:0T
batch_normalization_3/beta:04batch_normalization_3/cond/ReadVariableOp_3/Switch:0j
#batch_normalization_3/moving_mean:0Cbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0p
'batch_normalization_3/moving_variance:0Ebatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0K
activation_3/Relu:04batch_normalization_3/cond/FusedBatchNorm_1/Switch:0L
$batch_normalization_3/cond/pred_id:0$batch_normalization_3/cond/pred_id:0U
batch_normalization_3/gamma:04batch_normalization_3/cond/ReadVariableOp_2/Switch:0
╟
&batch_normalization_3/cond_1/cond_text&batch_normalization_3/cond_1/pred_id:0'batch_normalization_3/cond_1/switch_t:0 *╔
$batch_normalization_3/cond_1/Const:0
&batch_normalization_3/cond_1/pred_id:0
'batch_normalization_3/cond_1/switch_t:0P
&batch_normalization_3/cond_1/pred_id:0&batch_normalization_3/cond_1/pred_id:0
╔
(batch_normalization_3/cond_1/cond_text_1&batch_normalization_3/cond_1/pred_id:0'batch_normalization_3/cond_1/switch_f:0*╦
&batch_normalization_3/cond_1/Const_1:0
&batch_normalization_3/cond_1/pred_id:0
'batch_normalization_3/cond_1/switch_f:0P
&batch_normalization_3/cond_1/pred_id:0&batch_normalization_3/cond_1/pred_id:0
А	
$batch_normalization_4/cond/cond_text$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_t:0 *И
activation_4/Relu:0
batch_normalization_4/beta:0
"batch_normalization_4/cond/Const:0
$batch_normalization_4/cond/Const_1:0
2batch_normalization_4/cond/FusedBatchNorm/Switch:1
+batch_normalization_4/cond/FusedBatchNorm:0
+batch_normalization_4/cond/FusedBatchNorm:1
+batch_normalization_4/cond/FusedBatchNorm:2
+batch_normalization_4/cond/FusedBatchNorm:3
+batch_normalization_4/cond/FusedBatchNorm:4
2batch_normalization_4/cond/ReadVariableOp/Switch:1
+batch_normalization_4/cond/ReadVariableOp:0
4batch_normalization_4/cond/ReadVariableOp_1/Switch:1
-batch_normalization_4/cond/ReadVariableOp_1:0
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_t:0
batch_normalization_4/gamma:0I
activation_4/Relu:02batch_normalization_4/cond/FusedBatchNorm/Switch:1S
batch_normalization_4/gamma:02batch_normalization_4/cond/ReadVariableOp/Switch:1T
batch_normalization_4/beta:04batch_normalization_4/cond/ReadVariableOp_1/Switch:1L
$batch_normalization_4/cond/pred_id:0$batch_normalization_4/cond/pred_id:0
А
&batch_normalization_4/cond/cond_text_1$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_f:0*И
activation_4/Relu:0
batch_normalization_4/beta:0
Cbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_4/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_4/cond/FusedBatchNorm_1:0
-batch_normalization_4/cond/FusedBatchNorm_1:1
-batch_normalization_4/cond/FusedBatchNorm_1:2
-batch_normalization_4/cond/FusedBatchNorm_1:3
-batch_normalization_4/cond/FusedBatchNorm_1:4
4batch_normalization_4/cond/ReadVariableOp_2/Switch:0
-batch_normalization_4/cond/ReadVariableOp_2:0
4batch_normalization_4/cond/ReadVariableOp_3/Switch:0
-batch_normalization_4/cond/ReadVariableOp_3:0
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_f:0
batch_normalization_4/gamma:0
#batch_normalization_4/moving_mean:0
'batch_normalization_4/moving_variance:0U
batch_normalization_4/gamma:04batch_normalization_4/cond/ReadVariableOp_2/Switch:0p
'batch_normalization_4/moving_variance:0Ebatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0T
batch_normalization_4/beta:04batch_normalization_4/cond/ReadVariableOp_3/Switch:0j
#batch_normalization_4/moving_mean:0Cbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0K
activation_4/Relu:04batch_normalization_4/cond/FusedBatchNorm_1/Switch:0L
$batch_normalization_4/cond/pred_id:0$batch_normalization_4/cond/pred_id:0
╟
&batch_normalization_4/cond_1/cond_text&batch_normalization_4/cond_1/pred_id:0'batch_normalization_4/cond_1/switch_t:0 *╔
$batch_normalization_4/cond_1/Const:0
&batch_normalization_4/cond_1/pred_id:0
'batch_normalization_4/cond_1/switch_t:0P
&batch_normalization_4/cond_1/pred_id:0&batch_normalization_4/cond_1/pred_id:0
╔
(batch_normalization_4/cond_1/cond_text_1&batch_normalization_4/cond_1/pred_id:0'batch_normalization_4/cond_1/switch_f:0*╦
&batch_normalization_4/cond_1/Const_1:0
&batch_normalization_4/cond_1/pred_id:0
'batch_normalization_4/cond_1/switch_f:0P
&batch_normalization_4/cond_1/pred_id:0&batch_normalization_4/cond_1/pred_id:0"╞1
	variables╕1╡1
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
k
conv2d/bias:0conv2d/bias/Assign!conv2d/bias/Read/ReadVariableOp:0(2conv2d/bias/Initializer/zeros:08
в
batch_normalization/gamma:0 batch_normalization/gamma/Assign/batch_normalization/gamma/Read/ReadVariableOp:0(2,batch_normalization/gamma/Initializer/ones:08
Я
batch_normalization/beta:0batch_normalization/beta/Assign.batch_normalization/beta/Read/ReadVariableOp:0(2,batch_normalization/beta/Initializer/zeros:08
╜
!batch_normalization/moving_mean:0&batch_normalization/moving_mean/Assign5batch_normalization/moving_mean/Read/ReadVariableOp:0(23batch_normalization/moving_mean/Initializer/zeros:0@H
╠
%batch_normalization/moving_variance:0*batch_normalization/moving_variance/Assign9batch_normalization/moving_variance/Read/ReadVariableOp:0(26batch_normalization/moving_variance/Initializer/ones:0@H
Д
conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08
к
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
з
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
┼
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign7batch_normalization_1/moving_mean/Read/ReadVariableOp:0(25batch_normalization_1/moving_mean/Initializer/zeros:0@H
╘
'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign;batch_normalization_1/moving_variance/Read/ReadVariableOp:0(28batch_normalization_1/moving_variance/Initializer/ones:0@H
Д
conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08
к
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2.batch_normalization_2/gamma/Initializer/ones:08
з
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2.batch_normalization_2/beta/Initializer/zeros:08
┼
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign7batch_normalization_2/moving_mean/Read/ReadVariableOp:0(25batch_normalization_2/moving_mean/Initializer/zeros:0@H
╘
'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign;batch_normalization_2/moving_variance/Read/ReadVariableOp:0(28batch_normalization_2/moving_variance/Initializer/ones:0@H
Д
conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2,conv2d_3/kernel/Initializer/random_uniform:08
s
conv2d_3/bias:0conv2d_3/bias/Assign#conv2d_3/bias/Read/ReadVariableOp:0(2!conv2d_3/bias/Initializer/zeros:08
к
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign1batch_normalization_3/gamma/Read/ReadVariableOp:0(2.batch_normalization_3/gamma/Initializer/ones:08
з
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign0batch_normalization_3/beta/Read/ReadVariableOp:0(2.batch_normalization_3/beta/Initializer/zeros:08
┼
#batch_normalization_3/moving_mean:0(batch_normalization_3/moving_mean/Assign7batch_normalization_3/moving_mean/Read/ReadVariableOp:0(25batch_normalization_3/moving_mean/Initializer/zeros:0@H
╘
'batch_normalization_3/moving_variance:0,batch_normalization_3/moving_variance/Assign;batch_normalization_3/moving_variance/Read/ReadVariableOp:0(28batch_normalization_3/moving_variance/Initializer/ones:0@H
Д
conv2d_4/kernel:0conv2d_4/kernel/Assign%conv2d_4/kernel/Read/ReadVariableOp:0(2,conv2d_4/kernel/Initializer/random_uniform:08
s
conv2d_4/bias:0conv2d_4/bias/Assign#conv2d_4/bias/Read/ReadVariableOp:0(2!conv2d_4/bias/Initializer/zeros:08
к
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign1batch_normalization_4/gamma/Read/ReadVariableOp:0(2.batch_normalization_4/gamma/Initializer/ones:08
з
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign0batch_normalization_4/beta/Read/ReadVariableOp:0(2.batch_normalization_4/beta/Initializer/zeros:08
┼
#batch_normalization_4/moving_mean:0(batch_normalization_4/moving_mean/Assign7batch_normalization_4/moving_mean/Read/ReadVariableOp:0(25batch_normalization_4/moving_mean/Initializer/zeros:0@H
╘
'batch_normalization_4/moving_variance:0,batch_normalization_4/moving_variance/Assign;batch_normalization_4/moving_variance/Read/ReadVariableOp:0(28batch_normalization_4/moving_variance/Initializer/ones:0@H
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
М
prediction/kernel:0prediction/kernel/Assign'prediction/kernel/Read/ReadVariableOp:0(2.prediction/kernel/Initializer/random_uniform:08
{
prediction/bias:0prediction/bias/Assign%prediction/bias/Read/ReadVariableOp:0(2#prediction/bias/Initializer/zeros:08
_

SGD/iter:0SGD/iter/AssignSGD/iter/Read/ReadVariableOp:0(2SGD/iter/Initializer/zeros:0H
k
SGD/decay:0SGD/decay/AssignSGD/decay/Read/ReadVariableOp:0(2%SGD/decay/Initializer/initial_value:0H
Л
SGD/learning_rate:0SGD/learning_rate/Assign'SGD/learning_rate/Read/ReadVariableOp:0(2-SGD/learning_rate/Initializer/initial_value:0H
w
SGD/momentum:0SGD/momentum/Assign"SGD/momentum/Read/ReadVariableOp:0(2(SGD/momentum/Initializer/initial_value:0H
╡
 SGD/prediction/kernel/momentum:0%SGD/prediction/kernel/momentum/Assign4SGD/prediction/kernel/momentum/Read/ReadVariableOp:0(22SGD/prediction/kernel/momentum/Initializer/zeros:0
н
SGD/prediction/bias/momentum:0#SGD/prediction/bias/momentum/Assign2SGD/prediction/bias/momentum/Read/ReadVariableOp:0(20SGD/prediction/bias/momentum/Initializer/zeros:0"И
trainable_variablesЁэ
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
k
conv2d/bias:0conv2d/bias/Assign!conv2d/bias/Read/ReadVariableOp:0(2conv2d/bias/Initializer/zeros:08
в
batch_normalization/gamma:0 batch_normalization/gamma/Assign/batch_normalization/gamma/Read/ReadVariableOp:0(2,batch_normalization/gamma/Initializer/ones:08
Я
batch_normalization/beta:0batch_normalization/beta/Assign.batch_normalization/beta/Read/ReadVariableOp:0(2,batch_normalization/beta/Initializer/zeros:08
Д
conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08
к
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
з
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
Д
conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08
к
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2.batch_normalization_2/gamma/Initializer/ones:08
з
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2.batch_normalization_2/beta/Initializer/zeros:08
Д
conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2,conv2d_3/kernel/Initializer/random_uniform:08
s
conv2d_3/bias:0conv2d_3/bias/Assign#conv2d_3/bias/Read/ReadVariableOp:0(2!conv2d_3/bias/Initializer/zeros:08
к
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign1batch_normalization_3/gamma/Read/ReadVariableOp:0(2.batch_normalization_3/gamma/Initializer/ones:08
з
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign0batch_normalization_3/beta/Read/ReadVariableOp:0(2.batch_normalization_3/beta/Initializer/zeros:08
Д
conv2d_4/kernel:0conv2d_4/kernel/Assign%conv2d_4/kernel/Read/ReadVariableOp:0(2,conv2d_4/kernel/Initializer/random_uniform:08
s
conv2d_4/bias:0conv2d_4/bias/Assign#conv2d_4/bias/Read/ReadVariableOp:0(2!conv2d_4/bias/Initializer/zeros:08
к
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign1batch_normalization_4/gamma/Read/ReadVariableOp:0(2.batch_normalization_4/gamma/Initializer/ones:08
з
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign0batch_normalization_4/beta/Read/ReadVariableOp:0(2.batch_normalization_4/beta/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
М
prediction/kernel:0prediction/kernel/Assign'prediction/kernel/Read/ReadVariableOp:0(2.prediction/kernel/Initializer/random_uniform:08
{
prediction/bias:0prediction/bias/Assign%prediction/bias/Read/ReadVariableOp:0(2#prediction/bias/Initializer/zeros:08*░
serving_defaultЬ
9
input_image*
	input_1:0         ААC
prediction/Softmax:0+
prediction/Softmax:0         tensorflow/serving/predict
ªF
êÍ
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring "serve*2.15.02unknown8æ5

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*E
value<B: B4

f

signatures* 

trace_0* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *%
f R
__inference__traced_save_180

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_restore_189Â/

i
__inference__traced_save_180
file_prefix
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHo
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B Ø
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:³
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 7
NoOpNoOp^MergeV2Checkpoints*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:=9

_output_shapes
: 

_user_specified_nameConst
þ
A
 __inference_converted_fun_tf_142
	args_tf_0

identity_1D
jax2tf_arg_0Identity	args_tf_0*
T0*
_output_shapes
: v
 jax2tf_wrapped_fun_/pjit_fn_/MulMuljax2tf_arg_0:output:0jax2tf_arg_0:output:0*
T0*
_output_shapes
: 
"jax2tf_wrapped_fun_/pjit_fn_/Mul_1Mul$jax2tf_wrapped_fun_/pjit_fn_/Mul:z:0jax2tf_arg_0:output:0*
T0*
_output_shapes
: ]
IdentityIdentity&jax2tf_wrapped_fun_/pjit_fn_/Mul_1:z:0*
T0*
_output_shapes
: §
	IdentityN	IdentityN&jax2tf_wrapped_fun_/pjit_fn_/Mul_1:z:0jax2tf_arg_0:output:0*
T
2*)
_gradient_op_typeCustomGradient-134*
_output_shapes
: : K

jax2tf_outIdentityIdentityN:output:0*
T0*
_output_shapes
: L

Identity_1Identityjax2tf_out:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
: *
	_noinline(:A =

_output_shapes
: 
#
_user_specified_name	args_tf_0

E
__inference__traced_restore_189
file_prefix

identity_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHr
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B £
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
2Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 X
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: J

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
æ

 __inference_internal_grad_fn_161
result_grads_0
result_grads_1(
$jax2tf_vjp_jax2tf_arg_0_jax2tf_arg_0
identityj
jax2tf_vjp/jax2tf_arg_0Identity$jax2tf_vjp_jax2tf_arg_0_jax2tf_arg_0*
T0*
_output_shapes
: T
jax2tf_vjp/jax2tf_arg_1Identityresult_grads_0*
T0*
_output_shapes
: 
/jax2tf_vjp/jax2tf_fun_vjp_jax_/jvp/pjit_fn_/MulMul jax2tf_vjp/jax2tf_arg_0:output:0 jax2tf_vjp/jax2tf_arg_0:output:0*
T0*
_output_shapes
: °
1jax2tf_vjp/jax2tf_fun_vjp_jax_/jvp/pjit_fn_/Mul_1Mul3jax2tf_vjp/jax2tf_fun_vjp_jax_/jvp/pjit_fn_/Mul:z:0 jax2tf_vjp/jax2tf_arg_0:output:0*
T0*
_output_shapes
: U
jax2tf_vjp/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ÈB
0jax2tf_vjp/jax2tf_fun_vjp_jax_/transpose/jvp/MulMul jax2tf_vjp/jax2tf_arg_1:output:0jax2tf_vjp/Const:output:0*
T0*
_output_shapes
: v
jax2tf_vjp/IdentityIdentity4jax2tf_vjp/jax2tf_fun_vjp_jax_/transpose/jvp/Mul:z:0*
T0*
_output_shapes
: ð
jax2tf_vjp/IdentityN	IdentityN4jax2tf_vjp/jax2tf_fun_vjp_jax_/transpose/jvp/Mul:z:0 jax2tf_vjp/jax2tf_arg_0:output:0 jax2tf_vjp/jax2tf_arg_1:output:0*
T
2*)
_gradient_op_typeCustomGradient-150*
_output_shapes
: : : a
jax2tf_vjp/jax2tf_outIdentityjax2tf_vjp/IdentityN:output:0*
T0*
_output_shapes
: U
IdentityIdentityjax2tf_vjp/jax2tf_out:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:D@

_output_shapes
: 
&
_user_specified_namejax2tf_arg_06
 __inference_internal_grad_fn_161CustomGradient-134"ÞJ
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:¤
5
f

signatures"
_generic_user_object
â
trace_02Å
 __inference_converted_fun_tf_142 
²
FullArgSpec
args 
varargs	jargs_tf
varkwj	kwargs_tf
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0ztrace_0
"
signature_map
ÐBÍ
 __inference_converted_fun_tf_142	args_tf_0"
²
FullArgSpec
args
j	args_tf_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
4b2
jax2tf_arg_0:0 __inference_converted_fun_tf_142Y
 __inference_converted_fun_tf_1425!¢
¢

	args_tf_0 
ª "
unknown 
 __inference_internal_grad_fn_161dC¢@
9¢6

 

result_grads_0 

result_grads_1 
ª "

 

tensor_1 
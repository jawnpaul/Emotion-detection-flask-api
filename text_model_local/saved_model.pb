ъљ

е§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
Й
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
executor_typestring ѕ
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ"serve*2.0.02v2.0.0-rc2-26-g64c3d388Ю┬
Д
$text_model_2/embedding_29/embeddingsVarHandleOp*
shape:║Ь╚*5
shared_name&$text_model_2/embedding_29/embeddings*
dtype0*
_output_shapes
: 
а
8text_model_2/embedding_29/embeddings/Read/ReadVariableOpReadVariableOp$text_model_2/embedding_29/embeddings*!
_output_shapes
:║Ь╚*
dtype0
Ћ
text_model_2/conv1d/kernelVarHandleOp*+
shared_nametext_model_2/conv1d/kernel*
dtype0*
_output_shapes
: *
shape:╚d
ј
.text_model_2/conv1d/kernel/Read/ReadVariableOpReadVariableOptext_model_2/conv1d/kernel*
dtype0*#
_output_shapes
:╚d
ѕ
text_model_2/conv1d/biasVarHandleOp*
_output_shapes
: *
shape:d*)
shared_nametext_model_2/conv1d/bias*
dtype0
Ђ
,text_model_2/conv1d/bias/Read/ReadVariableOpReadVariableOptext_model_2/conv1d/bias*
dtype0*
_output_shapes
:d
Ў
text_model_2/conv1d_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:╚d*-
shared_nametext_model_2/conv1d_1/kernel
њ
0text_model_2/conv1d_1/kernel/Read/ReadVariableOpReadVariableOptext_model_2/conv1d_1/kernel*
dtype0*#
_output_shapes
:╚d
ї
text_model_2/conv1d_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:d*+
shared_nametext_model_2/conv1d_1/bias
Ё
.text_model_2/conv1d_1/bias/Read/ReadVariableOpReadVariableOptext_model_2/conv1d_1/bias*
dtype0*
_output_shapes
:d
Ў
text_model_2/conv1d_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:╚d*-
shared_nametext_model_2/conv1d_2/kernel
њ
0text_model_2/conv1d_2/kernel/Read/ReadVariableOpReadVariableOptext_model_2/conv1d_2/kernel*
dtype0*#
_output_shapes
:╚d
ї
text_model_2/conv1d_2/biasVarHandleOp*
shape:d*+
shared_nametext_model_2/conv1d_2/bias*
dtype0*
_output_shapes
: 
Ё
.text_model_2/conv1d_2/bias/Read/ReadVariableOpReadVariableOptext_model_2/conv1d_2/bias*
_output_shapes
:d*
dtype0
ќ
text_model_2/dense_48/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
гђ*-
shared_nametext_model_2/dense_48/kernel
Ј
0text_model_2/dense_48/kernel/Read/ReadVariableOpReadVariableOptext_model_2/dense_48/kernel*
dtype0* 
_output_shapes
:
гђ
Ї
text_model_2/dense_48/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*+
shared_nametext_model_2/dense_48/bias
є
.text_model_2/dense_48/bias/Read/ReadVariableOpReadVariableOptext_model_2/dense_48/bias*
dtype0*
_output_shapes	
:ђ
Ћ
text_model_2/dense_49/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	ђ*-
shared_nametext_model_2/dense_49/kernel
ј
0text_model_2/dense_49/kernel/Read/ReadVariableOpReadVariableOptext_model_2/dense_49/kernel*
dtype0*
_output_shapes
:	ђ
ї
text_model_2/dense_49/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*+
shared_nametext_model_2/dense_49/bias
Ё
.text_model_2/dense_49/bias/Read/ReadVariableOpReadVariableOptext_model_2/dense_49/bias*
dtype0*
_output_shapes
:
l
RMSprop/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
dtype0	*
_output_shapes
: 
n
RMSprop/decayVarHandleOp*
shape: *
shared_nameRMSprop/decay*
dtype0*
_output_shapes
: 
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
dtype0*
_output_shapes
: 
~
RMSprop/learning_rateVarHandleOp*&
shared_nameRMSprop/learning_rate*
dtype0*
_output_shapes
: *
shape: 
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
dtype0*
_output_shapes
: 
t
RMSprop/momentumVarHandleOp*
dtype0*
_output_shapes
: *
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
dtype0*
_output_shapes
: 
j
RMSprop/rhoVarHandleOp*
shape: *
shared_nameRMSprop/rho*
dtype0*
_output_shapes
: 
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shared_namecount*
dtype0*
_output_shapes
: *
shape: 
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
┐
0RMSprop/text_model_2/embedding_29/embeddings/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:║Ь╚*A
shared_name20RMSprop/text_model_2/embedding_29/embeddings/rms
И
DRMSprop/text_model_2/embedding_29/embeddings/rms/Read/ReadVariableOpReadVariableOp0RMSprop/text_model_2/embedding_29/embeddings/rms*
dtype0*!
_output_shapes
:║Ь╚
Г
&RMSprop/text_model_2/conv1d/kernel/rmsVarHandleOp*7
shared_name(&RMSprop/text_model_2/conv1d/kernel/rms*
dtype0*
_output_shapes
: *
shape:╚d
д
:RMSprop/text_model_2/conv1d/kernel/rms/Read/ReadVariableOpReadVariableOp&RMSprop/text_model_2/conv1d/kernel/rms*
dtype0*#
_output_shapes
:╚d
а
$RMSprop/text_model_2/conv1d/bias/rmsVarHandleOp*5
shared_name&$RMSprop/text_model_2/conv1d/bias/rms*
dtype0*
_output_shapes
: *
shape:d
Ў
8RMSprop/text_model_2/conv1d/bias/rms/Read/ReadVariableOpReadVariableOp$RMSprop/text_model_2/conv1d/bias/rms*
dtype0*
_output_shapes
:d
▒
(RMSprop/text_model_2/conv1d_1/kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:╚d*9
shared_name*(RMSprop/text_model_2/conv1d_1/kernel/rms
ф
<RMSprop/text_model_2/conv1d_1/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/text_model_2/conv1d_1/kernel/rms*
dtype0*#
_output_shapes
:╚d
ц
&RMSprop/text_model_2/conv1d_1/bias/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:d*7
shared_name(&RMSprop/text_model_2/conv1d_1/bias/rms
Ю
:RMSprop/text_model_2/conv1d_1/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/text_model_2/conv1d_1/bias/rms*
dtype0*
_output_shapes
:d
▒
(RMSprop/text_model_2/conv1d_2/kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:╚d*9
shared_name*(RMSprop/text_model_2/conv1d_2/kernel/rms
ф
<RMSprop/text_model_2/conv1d_2/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/text_model_2/conv1d_2/kernel/rms*
dtype0*#
_output_shapes
:╚d
ц
&RMSprop/text_model_2/conv1d_2/bias/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:d*7
shared_name(&RMSprop/text_model_2/conv1d_2/bias/rms
Ю
:RMSprop/text_model_2/conv1d_2/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/text_model_2/conv1d_2/bias/rms*
dtype0*
_output_shapes
:d
«
(RMSprop/text_model_2/dense_48/kernel/rmsVarHandleOp*
shape:
гђ*9
shared_name*(RMSprop/text_model_2/dense_48/kernel/rms*
dtype0*
_output_shapes
: 
Д
<RMSprop/text_model_2/dense_48/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/text_model_2/dense_48/kernel/rms*
dtype0* 
_output_shapes
:
гђ
Ц
&RMSprop/text_model_2/dense_48/bias/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*7
shared_name(&RMSprop/text_model_2/dense_48/bias/rms
ъ
:RMSprop/text_model_2/dense_48/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/text_model_2/dense_48/bias/rms*
dtype0*
_output_shapes	
:ђ
Г
(RMSprop/text_model_2/dense_49/kernel/rmsVarHandleOp*9
shared_name*(RMSprop/text_model_2/dense_49/kernel/rms*
dtype0*
_output_shapes
: *
shape:	ђ
д
<RMSprop/text_model_2/dense_49/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/text_model_2/dense_49/kernel/rms*
dtype0*
_output_shapes
:	ђ
ц
&RMSprop/text_model_2/dense_49/bias/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:*7
shared_name(&RMSprop/text_model_2/dense_49/bias/rms
Ю
:RMSprop/text_model_2/dense_49/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/text_model_2/dense_49/bias/rms*
dtype0*
_output_shapes
:

NoOpNoOp
љ4
ConstConst"/device:CPU:0*╦3
value┴3BЙ3 Bи3
С
	embedding

cnn_layer1

cnn_layer2

cnn_layer3
pool
dense_1
dropout

last_dense
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
b

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
R
&regularization_losses
'	variables
(trainable_variables
)	keras_api
h

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
R
0regularization_losses
1	variables
2trainable_variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
И
:iter
	;decay
<learning_rate
=momentum
>rho	rmso	rmsp	rmsq	rmsr	rmss	 rmst	!rmsu	*rmsv	+rmsw	4rmsx	5rmsy
 
N
0
1
2
3
4
 5
!6
*7
+8
49
510
N
0
1
2
3
4
 5
!6
*7
+8
49
510
џ
?layer_regularization_losses

regularization_losses
@metrics
	variables
trainable_variables
Anon_trainable_variables

Blayers
 
ig
VARIABLE_VALUE$text_model_2/embedding_29/embeddings/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
џ
regularization_losses
Clayer_regularization_losses
Dmetrics
	variables
trainable_variables
Enon_trainable_variables

Flayers
\Z
VARIABLE_VALUEtext_model_2/conv1d/kernel,cnn_layer1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEtext_model_2/conv1d/bias*cnn_layer1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
џ
regularization_losses
Glayer_regularization_losses
Hmetrics
	variables
trainable_variables
Inon_trainable_variables

Jlayers
^\
VARIABLE_VALUEtext_model_2/conv1d_1/kernel,cnn_layer2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEtext_model_2/conv1d_1/bias*cnn_layer2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
џ
regularization_losses
Klayer_regularization_losses
Lmetrics
	variables
trainable_variables
Mnon_trainable_variables

Nlayers
^\
VARIABLE_VALUEtext_model_2/conv1d_2/kernel,cnn_layer3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEtext_model_2/conv1d_2/bias*cnn_layer3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
џ
"regularization_losses
Olayer_regularization_losses
Pmetrics
#	variables
$trainable_variables
Qnon_trainable_variables

Rlayers
 
 
 
џ
&regularization_losses
Slayer_regularization_losses
Tmetrics
'	variables
(trainable_variables
Unon_trainable_variables

Vlayers
[Y
VARIABLE_VALUEtext_model_2/dense_48/kernel)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtext_model_2/dense_48/bias'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1

*0
+1
џ
,regularization_losses
Wlayer_regularization_losses
Xmetrics
-	variables
.trainable_variables
Ynon_trainable_variables

Zlayers
 
 
 
џ
0regularization_losses
[layer_regularization_losses
\metrics
1	variables
2trainable_variables
]non_trainable_variables

^layers
^\
VARIABLE_VALUEtext_model_2/dense_49/kernel,last_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEtext_model_2/dense_49/bias*last_dense/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
џ
6regularization_losses
_layer_regularization_losses
`metrics
7	variables
8trainable_variables
anon_trainable_variables

blayers
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
 

c0
 
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	dtotal
	ecount
f
_fn_kwargs
gregularization_losses
h	variables
itrainable_variables
j	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

d0
e1
 
џ
gregularization_losses
klayer_regularization_losses
lmetrics
h	variables
itrainable_variables
mnon_trainable_variables

nlayers
 
 

d0
e1
 
ћЉ
VARIABLE_VALUE0RMSprop/text_model_2/embedding_29/embeddings/rmsMembedding/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Єё
VARIABLE_VALUE&RMSprop/text_model_2/conv1d/kernel/rmsJcnn_layer1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUE$RMSprop/text_model_2/conv1d/bias/rmsHcnn_layer1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE(RMSprop/text_model_2/conv1d_1/kernel/rmsJcnn_layer2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE&RMSprop/text_model_2/conv1d_1/bias/rmsHcnn_layer2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE(RMSprop/text_model_2/conv1d_2/kernel/rmsJcnn_layer3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE&RMSprop/text_model_2/conv1d_2/bias/rmsHcnn_layer3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUE(RMSprop/text_model_2/dense_48/kernel/rmsGdense_1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUE&RMSprop/text_model_2/dense_48/bias/rmsEdense_1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE(RMSprop/text_model_2/dense_49/kernel/rmsJlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE&RMSprop/text_model_2/dense_49/bias/rmsHlast_dense/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
z
serving_default_input_1Placeholder*
dtype0*'
_output_shapes
:         2*
shape:         2
Ж
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1$text_model_2/embedding_29/embeddingstext_model_2/conv1d/kerneltext_model_2/conv1d/biastext_model_2/conv1d_1/kerneltext_model_2/conv1d_1/biastext_model_2/conv1d_2/kerneltext_model_2/conv1d_2/biastext_model_2/dense_48/kerneltext_model_2/dense_48/biastext_model_2/dense_49/kerneltext_model_2/dense_49/bias**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         *-
_gradient_op_typePartitionedCall-582591*-
f(R&
$__inference_signature_wrapper_582255*
Tout
2
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
н
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename8text_model_2/embedding_29/embeddings/Read/ReadVariableOp.text_model_2/conv1d/kernel/Read/ReadVariableOp,text_model_2/conv1d/bias/Read/ReadVariableOp0text_model_2/conv1d_1/kernel/Read/ReadVariableOp.text_model_2/conv1d_1/bias/Read/ReadVariableOp0text_model_2/conv1d_2/kernel/Read/ReadVariableOp.text_model_2/conv1d_2/bias/Read/ReadVariableOp0text_model_2/dense_48/kernel/Read/ReadVariableOp.text_model_2/dense_48/bias/Read/ReadVariableOp0text_model_2/dense_49/kernel/Read/ReadVariableOp.text_model_2/dense_49/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpDRMSprop/text_model_2/embedding_29/embeddings/rms/Read/ReadVariableOp:RMSprop/text_model_2/conv1d/kernel/rms/Read/ReadVariableOp8RMSprop/text_model_2/conv1d/bias/rms/Read/ReadVariableOp<RMSprop/text_model_2/conv1d_1/kernel/rms/Read/ReadVariableOp:RMSprop/text_model_2/conv1d_1/bias/rms/Read/ReadVariableOp<RMSprop/text_model_2/conv1d_2/kernel/rms/Read/ReadVariableOp:RMSprop/text_model_2/conv1d_2/bias/rms/Read/ReadVariableOp<RMSprop/text_model_2/dense_48/kernel/rms/Read/ReadVariableOp:RMSprop/text_model_2/dense_48/bias/rms/Read/ReadVariableOp<RMSprop/text_model_2/dense_49/kernel/rms/Read/ReadVariableOp:RMSprop/text_model_2/dense_49/bias/rms/Read/ReadVariableOpConst*
Tout
2**
config_proto

CPU

GPU 2J 8*
_output_shapes
: **
Tin#
!2	*-
_gradient_op_typePartitionedCall-582642*(
f#R!
__inference__traced_save_582641
І	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename$text_model_2/embedding_29/embeddingstext_model_2/conv1d/kerneltext_model_2/conv1d/biastext_model_2/conv1d_1/kerneltext_model_2/conv1d_1/biastext_model_2/conv1d_2/kerneltext_model_2/conv1d_2/biastext_model_2/dense_48/kerneltext_model_2/dense_48/biastext_model_2/dense_49/kerneltext_model_2/dense_49/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcount0RMSprop/text_model_2/embedding_29/embeddings/rms&RMSprop/text_model_2/conv1d/kernel/rms$RMSprop/text_model_2/conv1d/bias/rms(RMSprop/text_model_2/conv1d_1/kernel/rms&RMSprop/text_model_2/conv1d_1/bias/rms(RMSprop/text_model_2/conv1d_2/kernel/rms&RMSprop/text_model_2/conv1d_2/bias/rms(RMSprop/text_model_2/dense_48/kernel/rms&RMSprop/text_model_2/dense_48/bias/rms(RMSprop/text_model_2/dense_49/kernel/rms&RMSprop/text_model_2/dense_49/bias/rms*-
_gradient_op_typePartitionedCall-582742*+
f&R$
"__inference__traced_restore_582741*
Tout
2**
config_proto

CPU

GPU 2J 8*)
Tin"
 2*
_output_shapes
: ┬«
эh
Н
F__inference_text_model_layer_call_and_return_conditional_losses_582341

inputs>
:embedding_29_embedding_lookup_read_readvariableop_resource6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource+
'dense_48_matmul_readvariableop_resource,
(dense_48_biasadd_readvariableop_resource+
'dense_49_matmul_readvariableop_resource,
(dense_49_biasadd_readvariableop_resource
identityѕбconv1d/BiasAdd/ReadVariableOpб)conv1d/conv1d/ExpandDims_1/ReadVariableOpбconv1d_1/BiasAdd/ReadVariableOpб+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpбconv1d_2/BiasAdd/ReadVariableOpб+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpбdense_48/BiasAdd/ReadVariableOpбdense_48/MatMul/ReadVariableOpбdense_49/BiasAdd/ReadVariableOpбdense_49/MatMul/ReadVariableOpбembedding_29/embedding_lookupб1embedding_29/embedding_lookup/Read/ReadVariableOpП
1embedding_29/embedding_lookup/Read/ReadVariableOpReadVariableOp:embedding_29_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*!
_output_shapes
:║Ь╚Ў
&embedding_29/embedding_lookup/IdentityIdentity9embedding_29/embedding_lookup/Read/ReadVariableOp:value:0*
T0*!
_output_shapes
:║Ь╚Т
embedding_29/embedding_lookupResourceGather:embedding_29_embedding_lookup_read_readvariableop_resourceinputs2^embedding_29/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@embedding_29/embedding_lookup/Read/ReadVariableOp*
Tindices0*
dtype0*,
_output_shapes
:         2╚Є
(embedding_29/embedding_lookup/Identity_1Identity&embedding_29/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*,
_output_shapes
:         2╚*
T0*D
_class:
86loc:@embedding_29/embedding_lookup/Read/ReadVariableOpъ
(embedding_29/embedding_lookup/Identity_2Identity1embedding_29/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         2╚^
conv1d/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: ╗
conv1d/conv1d/ExpandDims
ExpandDims1embedding_29/embedding_lookup/Identity_2:output:0%conv1d/conv1d/ExpandDims/dim:output:0*0
_output_shapes
:         2╚*
T0¤
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:╚d`
conv1d/conv1d/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : Х
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:╚d┬
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:         1dЁ
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:         1d*
squeeze_dims
«
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dќ
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         1db
conv1d/ReluReluconv1d/BiasAdd:output:0*+
_output_shapes
:         1d*
T0l
*global_max_pooling1d/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: А
global_max_pooling1d/MaxMaxconv1d/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         d`
conv1d_1/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: ┐
conv1d_1/conv1d/ExpandDims
ExpandDims1embedding_29/embedding_lookup/Identity_2:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         2╚М
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:╚db
 conv1d_1/conv1d/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : ╝
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:╚d╚
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
paddingVALID*/
_output_shapes
:         0d*
T0*
strides
Ѕ
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:         0d*
squeeze_dims
▓
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dю
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         0df
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:         0dn
,global_max_pooling1d_1/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: Д
global_max_pooling1d_1/MaxMaxconv1d_1/Relu:activations:05global_max_pooling1d_1/Max/reduction_indices:output:0*'
_output_shapes
:         d*
T0`
conv1d_2/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: ┐
conv1d_2/conv1d/ExpandDims
ExpandDims1embedding_29/embedding_lookup/Identity_2:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*0
_output_shapes
:         2╚*
T0М
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:╚db
 conv1d_2/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: ╝
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*'
_output_shapes
:╚d*
T0╚
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
paddingVALID*/
_output_shapes
:         /d*
T0*
strides
Ѕ
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*+
_output_shapes
:         /d*
squeeze_dims
*
T0▓
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dю
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         /df
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*+
_output_shapes
:         /d*
T0n
,global_max_pooling1d_2/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: Д
global_max_pooling1d_2/MaxMaxconv1d_2/Relu:activations:05global_max_pooling1d_2/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         dV
concat/axisConst*
_output_shapes
: *
valueB :
         *
dtype0Л
concatConcatV2!global_max_pooling1d/Max:output:0#global_max_pooling1d_1/Max:output:0#global_max_pooling1d_2/Max:output:0concat/axis:output:0*(
_output_shapes
:         г*
T0*
NХ
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
гђЁ
dense_48/MatMulMatMulconcat:output:0&dense_48/MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0│
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђњ
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђc
dense_48/ReluReludense_48/BiasAdd:output:0*
T0*(
_output_shapes
:         ђY
dropout/dropout/rateConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: `
dropout/dropout/ShapeShapedense_48/Relu:activations:0*
T0*
_output_shapes
:g
"dropout/dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    g
"dropout/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ю
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:         ђц
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ╗
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:         ђГ
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*(
_output_shapes
:         ђ*
T0Z
dropout/dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
_output_shapes
: *
T0^
dropout/dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: ђ
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: б
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*(
_output_shapes
:         ђ*
T0Є
dropout/dropout/mulMuldense_48/Relu:activations:0dropout/dropout/truediv:z:0*(
_output_shapes
:         ђ*
T0ђ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:         ђѓ
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђх
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђј
dense_49/MatMulMatMuldropout/dropout/mul_1:z:0&dense_49/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0▓
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Љ
dense_49/BiasAddBiasAdddense_49/MatMul:product:0'dense_49/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_49/SoftmaxSoftmaxdense_49/BiasAdd:output:0*'
_output_shapes
:         *
T0е
IdentityIdentitydense_49/Softmax:softmax:0^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp ^dense_49/BiasAdd/ReadVariableOp^dense_49/MatMul/ReadVariableOp^embedding_29/embedding_lookup2^embedding_29/embedding_lookup/Read/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*R
_input_shapesA
?:         2:::::::::::2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2f
1embedding_29/embedding_lookup/Read/ReadVariableOp1embedding_29/embedding_lookup/Read/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2>
embedding_29/embedding_lookupembedding_29/embedding_lookup2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : 
ш.
е
F__inference_text_model_layer_call_and_return_conditional_losses_582218

inputs/
+embedding_29_statefulpartitionedcall_args_1)
%conv1d_statefulpartitionedcall_args_1)
%conv1d_statefulpartitionedcall_args_2+
'conv1d_1_statefulpartitionedcall_args_1+
'conv1d_1_statefulpartitionedcall_args_2+
'conv1d_2_statefulpartitionedcall_args_1+
'conv1d_2_statefulpartitionedcall_args_2+
'dense_48_statefulpartitionedcall_args_1+
'dense_48_statefulpartitionedcall_args_2+
'dense_49_statefulpartitionedcall_args_1+
'dense_49_statefulpartitionedcall_args_2
identityѕбconv1d/StatefulPartitionedCallб conv1d_1/StatefulPartitionedCallб conv1d_2/StatefulPartitionedCallб dense_48/StatefulPartitionedCallб dense_49/StatefulPartitionedCallб$embedding_29/StatefulPartitionedCallЬ
$embedding_29/StatefulPartitionedCallStatefulPartitionedCallinputs+embedding_29_statefulpartitionedcall_args_1*Q
fLRJ
H__inference_embedding_29_layer_call_and_return_conditional_losses_581983*
Tout
2**
config_proto

CPU

GPU 2J 8*,
_output_shapes
:         2╚*
Tin
2*-
_gradient_op_typePartitionedCall-581989ф
conv1d/StatefulPartitionedCallStatefulPartitionedCall-embedding_29/StatefulPartitionedCall:output:0%conv1d_statefulpartitionedcall_args_1%conv1d_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-581884*K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_581878*
Tout
2**
config_proto

CPU

GPU 2J 8*+
_output_shapes
:         1d*
Tin
2▄
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         d*-
_gradient_op_typePartitionedCall-581964*Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958*
Tout
2▓
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall-embedding_29/StatefulPartitionedCall:output:0'conv1d_1_statefulpartitionedcall_args_1'conv1d_1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-581914*M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_581908*
Tout
2**
config_proto

CPU

GPU 2J 8*+
_output_shapes
:         0d*
Tin
2Я
&global_max_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-581964*Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         d▓
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall-embedding_29/StatefulPartitionedCall:output:0'conv1d_2_statefulpartitionedcall_args_1'conv1d_2_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*+
_output_shapes
:         /d*
Tin
2*-
_gradient_op_typePartitionedCall-581944*M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_581938*
Tout
2Я
&global_max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-581964*Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         dV
concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: ш
concatConcatV2-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d_1/PartitionedCall:output:0/global_max_pooling1d_2/PartitionedCall:output:0concat/axis:output:0*
T0*
N*(
_output_shapes
:         гЉ
 dense_48/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0'dense_48_statefulpartitionedcall_args_1'dense_48_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:         ђ*-
_gradient_op_typePartitionedCall-582029*M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_582023*
Tout
2┼
dropout/PartitionedCallPartitionedCall)dense_48/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:         ђ*
Tin
2*-
_gradient_op_typePartitionedCall-582079*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_582067А
 dense_49/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0'dense_49_statefulpartitionedcall_args_1'dense_49_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2*-
_gradient_op_typePartitionedCall-582101*M
fHRF
D__inference_dense_49_layer_call_and_return_conditional_losses_582095*
Tout
2┼
IdentityIdentity)dense_49/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*R
_input_shapesA
?:         2:::::::::::2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2L
$embedding_29/StatefulPartitionedCall$embedding_29/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall:	 :
 : :& "
 
_user_specified_nameinputs: : : : : : : : 
ц
э
D__inference_conv1d_1_layer_call_and_return_conditional_losses_581908

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpW
conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: І
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  ╚┴
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:╚dY
conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: А
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:╚dХ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*
strides
*
paddingVALID*8
_output_shapes&
$:"                  dђ
conv1d/SqueezeSqueezeconv1d:output:0*
squeeze_dims
*
T0*4
_output_shapes"
 :                  dа
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dі
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*4
_output_shapes"
 :                  d*
T0]
ReluReluBiasAdd:output:0*4
_output_shapes"
 :                  d*
T0Ц
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :                  d"
identityIdentity:output:0*<
_input_shapes+
):                  ╚::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Х
D
(__inference_dropout_layer_call_fn_582511

inputs
identityџ
PartitionedCallPartitionedCallinputs**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:         ђ*-
_gradient_op_typePartitionedCall-582079*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_582067*
Tout
2a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
є
a
C__inference_dropout_layer_call_and_return_conditional_losses_582067

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
║
a
(__inference_dropout_layer_call_fn_582506

inputs
identityѕбStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputs**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:         ђ*-
_gradient_op_typePartitionedCall-582071*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_582060*
Tout
2Ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
п
н
H__inference_embedding_29_layer_call_and_return_conditional_losses_581983

inputs1
-embedding_lookup_read_readvariableop_resource
identityѕбembedding_lookupб$embedding_lookup/Read/ReadVariableOp├
$embedding_lookup/Read/ReadVariableOpReadVariableOp-embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*!
_output_shapes
:║Ь╚
embedding_lookup/IdentityIdentity,embedding_lookup/Read/ReadVariableOp:value:0*
T0*!
_output_shapes
:║Ь╚▓
embedding_lookupResourceGather-embedding_lookup_read_readvariableop_resourceinputs%^embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*
dtype0*,
_output_shapes
:         2╚*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOpЯ
embedding_lookup/Identity_1Identityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         2╚ё
embedding_lookup/Identity_2Identity$embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         2╚Ф
IdentityIdentity$embedding_lookup/Identity_2:output:0^embedding_lookup%^embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         2╚*
T0"
identityIdentity:output:0**
_input_shapes
:         2:2L
$embedding_lookup/Read/ReadVariableOp$embedding_lookup/Read/ReadVariableOp2$
embedding_lookupembedding_lookup:& "
 
_user_specified_nameinputs: 
вh
«

!__inference__wrapped_model_581859
input_1I
Etext_model_embedding_29_embedding_lookup_read_readvariableop_resourceA
=text_model_conv1d_conv1d_expanddims_1_readvariableop_resource5
1text_model_conv1d_biasadd_readvariableop_resourceC
?text_model_conv1d_1_conv1d_expanddims_1_readvariableop_resource7
3text_model_conv1d_1_biasadd_readvariableop_resourceC
?text_model_conv1d_2_conv1d_expanddims_1_readvariableop_resource7
3text_model_conv1d_2_biasadd_readvariableop_resource6
2text_model_dense_48_matmul_readvariableop_resource7
3text_model_dense_48_biasadd_readvariableop_resource6
2text_model_dense_49_matmul_readvariableop_resource7
3text_model_dense_49_biasadd_readvariableop_resource
identityѕб(text_model/conv1d/BiasAdd/ReadVariableOpб4text_model/conv1d/conv1d/ExpandDims_1/ReadVariableOpб*text_model/conv1d_1/BiasAdd/ReadVariableOpб6text_model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpб*text_model/conv1d_2/BiasAdd/ReadVariableOpб6text_model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpб*text_model/dense_48/BiasAdd/ReadVariableOpб)text_model/dense_48/MatMul/ReadVariableOpб*text_model/dense_49/BiasAdd/ReadVariableOpб)text_model/dense_49/MatMul/ReadVariableOpб(text_model/embedding_29/embedding_lookupб<text_model/embedding_29/embedding_lookup/Read/ReadVariableOpз
<text_model/embedding_29/embedding_lookup/Read/ReadVariableOpReadVariableOpEtext_model_embedding_29_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*!
_output_shapes
:║Ь╚»
1text_model/embedding_29/embedding_lookup/IdentityIdentityDtext_model/embedding_29/embedding_lookup/Read/ReadVariableOp:value:0*
T0*!
_output_shapes
:║Ь╚Њ
(text_model/embedding_29/embedding_lookupResourceGatherEtext_model_embedding_29_embedding_lookup_read_readvariableop_resourceinput_1=^text_model/embedding_29/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*O
_classE
CAloc:@text_model/embedding_29/embedding_lookup/Read/ReadVariableOp*
Tindices0*
dtype0*,
_output_shapes
:         2╚е
3text_model/embedding_29/embedding_lookup/Identity_1Identity1text_model/embedding_29/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*O
_classE
CAloc:@text_model/embedding_29/embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         2╚*
T0┤
3text_model/embedding_29/embedding_lookup/Identity_2Identity<text_model/embedding_29/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         2╚i
'text_model/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
value	B :*
dtype0▄
#text_model/conv1d/conv1d/ExpandDims
ExpandDims<text_model/embedding_29/embedding_lookup/Identity_2:output:00text_model/conv1d/conv1d/ExpandDims/dim:output:0*0
_output_shapes
:         2╚*
T0т
4text_model/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=text_model_conv1d_conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:╚dk
)text_model/conv1d/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: О
%text_model/conv1d/conv1d/ExpandDims_1
ExpandDims<text_model/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:02text_model/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:╚dс
text_model/conv1d/conv1dConv2D,text_model/conv1d/conv1d/ExpandDims:output:0.text_model/conv1d/conv1d/ExpandDims_1:output:0*/
_output_shapes
:         1d*
T0*
strides
*
paddingVALIDЏ
 text_model/conv1d/conv1d/SqueezeSqueeze!text_model/conv1d/conv1d:output:0*
T0*+
_output_shapes
:         1d*
squeeze_dims
─
(text_model/conv1d/BiasAdd/ReadVariableOpReadVariableOp1text_model_conv1d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dи
text_model/conv1d/BiasAddBiasAdd)text_model/conv1d/conv1d/Squeeze:output:00text_model/conv1d/BiasAdd/ReadVariableOp:value:0*+
_output_shapes
:         1d*
T0x
text_model/conv1d/ReluRelu"text_model/conv1d/BiasAdd:output:0*+
_output_shapes
:         1d*
T0w
5text_model/global_max_pooling1d/Max/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :┬
#text_model/global_max_pooling1d/MaxMax$text_model/conv1d/Relu:activations:0>text_model/global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         dk
)text_model/conv1d_1/conv1d/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :Я
%text_model/conv1d_1/conv1d/ExpandDims
ExpandDims<text_model/embedding_29/embedding_lookup/Identity_2:output:02text_model/conv1d_1/conv1d/ExpandDims/dim:output:0*0
_output_shapes
:         2╚*
T0ж
6text_model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?text_model_conv1d_1_conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:╚dm
+text_model/conv1d_1/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: П
'text_model/conv1d_1/conv1d/ExpandDims_1
ExpandDims>text_model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:04text_model/conv1d_1/conv1d/ExpandDims_1/dim:output:0*'
_output_shapes
:╚d*
T0ж
text_model/conv1d_1/conv1dConv2D.text_model/conv1d_1/conv1d/ExpandDims:output:00text_model/conv1d_1/conv1d/ExpandDims_1:output:0*
paddingVALID*/
_output_shapes
:         0d*
T0*
strides
Ъ
"text_model/conv1d_1/conv1d/SqueezeSqueeze#text_model/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:         0d*
squeeze_dims
╚
*text_model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp3text_model_conv1d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dй
text_model/conv1d_1/BiasAddBiasAdd+text_model/conv1d_1/conv1d/Squeeze:output:02text_model/conv1d_1/BiasAdd/ReadVariableOp:value:0*+
_output_shapes
:         0d*
T0|
text_model/conv1d_1/ReluRelu$text_model/conv1d_1/BiasAdd:output:0*+
_output_shapes
:         0d*
T0y
7text_model/global_max_pooling1d_1/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: ╚
%text_model/global_max_pooling1d_1/MaxMax&text_model/conv1d_1/Relu:activations:0@text_model/global_max_pooling1d_1/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         dk
)text_model/conv1d_2/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: Я
%text_model/conv1d_2/conv1d/ExpandDims
ExpandDims<text_model/embedding_29/embedding_lookup/Identity_2:output:02text_model/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         2╚ж
6text_model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?text_model_conv1d_2_conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:╚dm
+text_model/conv1d_2/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: П
'text_model/conv1d_2/conv1d/ExpandDims_1
ExpandDims>text_model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:04text_model/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:╚dж
text_model/conv1d_2/conv1dConv2D.text_model/conv1d_2/conv1d/ExpandDims:output:00text_model/conv1d_2/conv1d/ExpandDims_1:output:0*
paddingVALID*/
_output_shapes
:         /d*
T0*
strides
Ъ
"text_model/conv1d_2/conv1d/SqueezeSqueeze#text_model/conv1d_2/conv1d:output:0*+
_output_shapes
:         /d*
squeeze_dims
*
T0╚
*text_model/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp3text_model_conv1d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dй
text_model/conv1d_2/BiasAddBiasAdd+text_model/conv1d_2/conv1d/Squeeze:output:02text_model/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         /d|
text_model/conv1d_2/ReluRelu$text_model/conv1d_2/BiasAdd:output:0*+
_output_shapes
:         /d*
T0y
7text_model/global_max_pooling1d_2/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: ╚
%text_model/global_max_pooling1d_2/MaxMax&text_model/conv1d_2/Relu:activations:0@text_model/global_max_pooling1d_2/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         da
text_model/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: ѕ
text_model/concatConcatV2,text_model/global_max_pooling1d/Max:output:0.text_model/global_max_pooling1d_1/Max:output:0.text_model/global_max_pooling1d_2/Max:output:0text_model/concat/axis:output:0*
T0*
N*(
_output_shapes
:         г╠
)text_model/dense_48/MatMul/ReadVariableOpReadVariableOp2text_model_dense_48_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
гђд
text_model/dense_48/MatMulMatMultext_model/concat:output:01text_model/dense_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ╔
*text_model/dense_48/BiasAdd/ReadVariableOpReadVariableOp3text_model_dense_48_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:ђ*
dtype0│
text_model/dense_48/BiasAddBiasAdd$text_model/dense_48/MatMul:product:02text_model/dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђy
text_model/dense_48/ReluRelu$text_model/dense_48/BiasAdd:output:0*
T0*(
_output_shapes
:         ђѓ
text_model/dropout/IdentityIdentity&text_model/dense_48/Relu:activations:0*(
_output_shapes
:         ђ*
T0╦
)text_model/dense_49/MatMul/ReadVariableOpReadVariableOp2text_model_dense_49_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђ»
text_model/dense_49/MatMulMatMul$text_model/dropout/Identity:output:01text_model/dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ╚
*text_model/dense_49/BiasAdd/ReadVariableOpReadVariableOp3text_model_dense_49_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:▓
text_model/dense_49/BiasAddBiasAdd$text_model/dense_49/MatMul:product:02text_model/dense_49/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ~
text_model/dense_49/SoftmaxSoftmax$text_model/dense_49/BiasAdd:output:0*
T0*'
_output_shapes
:         и
IdentityIdentity%text_model/dense_49/Softmax:softmax:0)^text_model/conv1d/BiasAdd/ReadVariableOp5^text_model/conv1d/conv1d/ExpandDims_1/ReadVariableOp+^text_model/conv1d_1/BiasAdd/ReadVariableOp7^text_model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+^text_model/conv1d_2/BiasAdd/ReadVariableOp7^text_model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+^text_model/dense_48/BiasAdd/ReadVariableOp*^text_model/dense_48/MatMul/ReadVariableOp+^text_model/dense_49/BiasAdd/ReadVariableOp*^text_model/dense_49/MatMul/ReadVariableOp)^text_model/embedding_29/embedding_lookup=^text_model/embedding_29/embedding_lookup/Read/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*R
_input_shapesA
?:         2:::::::::::2l
4text_model/conv1d/conv1d/ExpandDims_1/ReadVariableOp4text_model/conv1d/conv1d/ExpandDims_1/ReadVariableOp2V
)text_model/dense_49/MatMul/ReadVariableOp)text_model/dense_49/MatMul/ReadVariableOp2p
6text_model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp6text_model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2X
*text_model/dense_48/BiasAdd/ReadVariableOp*text_model/dense_48/BiasAdd/ReadVariableOp2p
6text_model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp6text_model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2|
<text_model/embedding_29/embedding_lookup/Read/ReadVariableOp<text_model/embedding_29/embedding_lookup/Read/ReadVariableOp2X
*text_model/conv1d_2/BiasAdd/ReadVariableOp*text_model/conv1d_2/BiasAdd/ReadVariableOp2X
*text_model/dense_49/BiasAdd/ReadVariableOp*text_model/dense_49/BiasAdd/ReadVariableOp2T
(text_model/conv1d/BiasAdd/ReadVariableOp(text_model/conv1d/BiasAdd/ReadVariableOp2T
(text_model/embedding_29/embedding_lookup(text_model/embedding_29/embedding_lookup2V
)text_model/dense_48/MatMul/ReadVariableOp)text_model/dense_48/MatMul/ReadVariableOp2X
*text_model/conv1d_1/BiasAdd/ReadVariableOp*text_model/conv1d_1/BiasAdd/ReadVariableOp:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : 
п
н
H__inference_embedding_29_layer_call_and_return_conditional_losses_582452

inputs1
-embedding_lookup_read_readvariableop_resource
identityѕбembedding_lookupб$embedding_lookup/Read/ReadVariableOp├
$embedding_lookup/Read/ReadVariableOpReadVariableOp-embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*!
_output_shapes
:║Ь╚
embedding_lookup/IdentityIdentity,embedding_lookup/Read/ReadVariableOp:value:0*
T0*!
_output_shapes
:║Ь╚▓
embedding_lookupResourceGather-embedding_lookup_read_readvariableop_resourceinputs%^embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*,
_output_shapes
:         2╚*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*
Tindices0Я
embedding_lookup/Identity_1Identityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         2╚ё
embedding_lookup/Identity_2Identity$embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         2╚Ф
IdentityIdentity$embedding_lookup/Identity_2:output:0^embedding_lookup%^embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         2╚*
T0"
identityIdentity:output:0**
_input_shapes
:         2:2L
$embedding_lookup/Read/ReadVariableOp$embedding_lookup/Read/ReadVariableOp2$
embedding_lookupembedding_lookup:& "
 
_user_specified_nameinputs: 
п	
П
D__inference_dense_48_layer_call_and_return_conditional_losses_582469

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpц
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
гђj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђw
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*/
_input_shapes
:         г::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
В?
Ы
__inference__traced_save_582641
file_prefixC
?savev2_text_model_2_embedding_29_embeddings_read_readvariableop9
5savev2_text_model_2_conv1d_kernel_read_readvariableop7
3savev2_text_model_2_conv1d_bias_read_readvariableop;
7savev2_text_model_2_conv1d_1_kernel_read_readvariableop9
5savev2_text_model_2_conv1d_1_bias_read_readvariableop;
7savev2_text_model_2_conv1d_2_kernel_read_readvariableop9
5savev2_text_model_2_conv1d_2_bias_read_readvariableop;
7savev2_text_model_2_dense_48_kernel_read_readvariableop9
5savev2_text_model_2_dense_48_bias_read_readvariableop;
7savev2_text_model_2_dense_49_kernel_read_readvariableop9
5savev2_text_model_2_dense_49_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopO
Ksavev2_rmsprop_text_model_2_embedding_29_embeddings_rms_read_readvariableopE
Asavev2_rmsprop_text_model_2_conv1d_kernel_rms_read_readvariableopC
?savev2_rmsprop_text_model_2_conv1d_bias_rms_read_readvariableopG
Csavev2_rmsprop_text_model_2_conv1d_1_kernel_rms_read_readvariableopE
Asavev2_rmsprop_text_model_2_conv1d_1_bias_rms_read_readvariableopG
Csavev2_rmsprop_text_model_2_conv1d_2_kernel_rms_read_readvariableopE
Asavev2_rmsprop_text_model_2_conv1d_2_bias_rms_read_readvariableopG
Csavev2_rmsprop_text_model_2_dense_48_kernel_rms_read_readvariableopE
Asavev2_rmsprop_text_model_2_dense_48_bias_rms_read_readvariableopG
Csavev2_rmsprop_text_model_2_dense_49_kernel_rms_read_readvariableopE
Asavev2_rmsprop_text_model_2_dense_49_bias_rms_read_readvariableop
savev2_1_const

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1ј
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_54cdf15d35aa4a76a8c56e9772a5f411/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Я
SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*Ѕ
value BЧB/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB,cnn_layer1/kernel/.ATTRIBUTES/VARIABLE_VALUEB*cnn_layer1/bias/.ATTRIBUTES/VARIABLE_VALUEB,cnn_layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB*cnn_layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB,cnn_layer3/kernel/.ATTRIBUTES/VARIABLE_VALUEB*cnn_layer3/bias/.ATTRIBUTES/VARIABLE_VALUEB)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUEB,last_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB*last_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBMembedding/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBJcnn_layer1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBJcnn_layer2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBJcnn_layer3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBGdense_1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEdense_1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBJlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBHlast_dense/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEД
SaveV2/shape_and_slicesConst"/device:CPU:0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:║
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0?savev2_text_model_2_embedding_29_embeddings_read_readvariableop5savev2_text_model_2_conv1d_kernel_read_readvariableop3savev2_text_model_2_conv1d_bias_read_readvariableop7savev2_text_model_2_conv1d_1_kernel_read_readvariableop5savev2_text_model_2_conv1d_1_bias_read_readvariableop7savev2_text_model_2_conv1d_2_kernel_read_readvariableop5savev2_text_model_2_conv1d_2_bias_read_readvariableop7savev2_text_model_2_dense_48_kernel_read_readvariableop5savev2_text_model_2_dense_48_bias_read_readvariableop7savev2_text_model_2_dense_49_kernel_read_readvariableop5savev2_text_model_2_dense_49_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopKsavev2_rmsprop_text_model_2_embedding_29_embeddings_rms_read_readvariableopAsavev2_rmsprop_text_model_2_conv1d_kernel_rms_read_readvariableop?savev2_rmsprop_text_model_2_conv1d_bias_rms_read_readvariableopCsavev2_rmsprop_text_model_2_conv1d_1_kernel_rms_read_readvariableopAsavev2_rmsprop_text_model_2_conv1d_1_bias_rms_read_readvariableopCsavev2_rmsprop_text_model_2_conv1d_2_kernel_rms_read_readvariableopAsavev2_rmsprop_text_model_2_conv1d_2_bias_rms_read_readvariableopCsavev2_rmsprop_text_model_2_dense_48_kernel_rms_read_readvariableopAsavev2_rmsprop_text_model_2_dense_48_bias_rms_read_readvariableopCsavev2_rmsprop_text_model_2_dense_49_kernel_rms_read_readvariableopAsavev2_rmsprop_text_model_2_dense_49_bias_rms_read_readvariableop"/device:CPU:0*
_output_shapes
 *+
dtypes!
2	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: Ќ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ѕ
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:├
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2╣
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
_output_shapes
:*
T0*
Nќ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*Є
_input_shapesш
Ы: :║Ь╚:╚d:d:╚d:d:╚d:d:
гђ:ђ:	ђ:: : : : : : : :║Ь╚:╚d:d:╚d:d:╚d:d:
гђ:ђ:	ђ:: 2
SaveV2_1SaveV2_12
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints: : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : :+ '
%
_user_specified_namefile_prefix: : 
ф
з
+__inference_text_model_layer_call_fn_582187
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11*O
fJRH
F__inference_text_model_layer_call_and_return_conditional_losses_582172*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2*-
_gradient_op_typePartitionedCall-582173ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*R
_input_shapesA
?:         2:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : 
░
b
C__inference_dropout_layer_call_and_return_conditional_losses_582060

inputs
identityѕQ
dropout/rateConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0_
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:         ђї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Б
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:         ђЋ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*(
_output_shapes
:         ђ*
T0R
dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ђ?*
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0і
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*(
_output_shapes
:         ђ*
T0b
dropout/mulMulinputsdropout/truediv:z:0*(
_output_shapes
:         ђ*
T0p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*(
_output_shapes
:         ђ*

SrcT0
j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђZ
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
Д
Ы
+__inference_text_model_layer_call_fn_582442

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identityѕбStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2*-
_gradient_op_typePartitionedCall-582219*O
fJRH
F__inference_text_model_layer_call_and_return_conditional_losses_582218*
Tout
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*R
_input_shapesA
?:         2:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : 
џ
l
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958

inputs
identityW
Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:                  ]
IdentityIdentityMax:output:0*0
_output_shapes
:                  *
T0"
identityIdentity:output:0*<
_input_shapes+
):'                           :& "
 
_user_specified_nameinputs
┘
ф
)__inference_dense_49_layer_call_fn_582529

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         *-
_gradient_op_typePartitionedCall-582101*M
fHRF
D__inference_dense_49_layer_call_and_return_conditional_losses_582095*
Tout
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
є
a
C__inference_dropout_layer_call_and_return_conditional_losses_582501

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:         ђ*
T0\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
Ьs
Ь
"__inference__traced_restore_582741
file_prefix9
5assignvariableop_text_model_2_embedding_29_embeddings1
-assignvariableop_1_text_model_2_conv1d_kernel/
+assignvariableop_2_text_model_2_conv1d_bias3
/assignvariableop_3_text_model_2_conv1d_1_kernel1
-assignvariableop_4_text_model_2_conv1d_1_bias3
/assignvariableop_5_text_model_2_conv1d_2_kernel1
-assignvariableop_6_text_model_2_conv1d_2_bias3
/assignvariableop_7_text_model_2_dense_48_kernel1
-assignvariableop_8_text_model_2_dense_48_bias3
/assignvariableop_9_text_model_2_dense_49_kernel2
.assignvariableop_10_text_model_2_dense_49_bias$
 assignvariableop_11_rmsprop_iter%
!assignvariableop_12_rmsprop_decay-
)assignvariableop_13_rmsprop_learning_rate(
$assignvariableop_14_rmsprop_momentum#
assignvariableop_15_rmsprop_rho
assignvariableop_16_total
assignvariableop_17_countH
Dassignvariableop_18_rmsprop_text_model_2_embedding_29_embeddings_rms>
:assignvariableop_19_rmsprop_text_model_2_conv1d_kernel_rms<
8assignvariableop_20_rmsprop_text_model_2_conv1d_bias_rms@
<assignvariableop_21_rmsprop_text_model_2_conv1d_1_kernel_rms>
:assignvariableop_22_rmsprop_text_model_2_conv1d_1_bias_rms@
<assignvariableop_23_rmsprop_text_model_2_conv1d_2_kernel_rms>
:assignvariableop_24_rmsprop_text_model_2_conv1d_2_bias_rms@
<assignvariableop_25_rmsprop_text_model_2_dense_48_kernel_rms>
:assignvariableop_26_rmsprop_text_model_2_dense_48_bias_rms@
<assignvariableop_27_rmsprop_text_model_2_dense_49_kernel_rms>
:assignvariableop_28_rmsprop_text_model_2_dense_49_bias_rms
identity_30ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9б	RestoreV2бRestoreV2_1с
RestoreV2/tensor_namesConst"/device:CPU:0*Ѕ
value BЧB/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB,cnn_layer1/kernel/.ATTRIBUTES/VARIABLE_VALUEB*cnn_layer1/bias/.ATTRIBUTES/VARIABLE_VALUEB,cnn_layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB*cnn_layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB,cnn_layer3/kernel/.ATTRIBUTES/VARIABLE_VALUEB*cnn_layer3/bias/.ATTRIBUTES/VARIABLE_VALUEB)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUEB,last_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB*last_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBMembedding/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBJcnn_layer1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBJcnn_layer2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBJcnn_layer3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBGdense_1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEdense_1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBJlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBHlast_dense/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:ф
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0░
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ѕ
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2	L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0Љ
AssignVariableOpAssignVariableOp5assignvariableop_text_model_2_embedding_29_embeddingsIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0Ї
AssignVariableOp_1AssignVariableOp-assignvariableop_1_text_model_2_conv1d_kernelIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0І
AssignVariableOp_2AssignVariableOp+assignvariableop_2_text_model_2_conv1d_biasIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:Ј
AssignVariableOp_3AssignVariableOp/assignvariableop_3_text_model_2_conv1d_1_kernelIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:Ї
AssignVariableOp_4AssignVariableOp-assignvariableop_4_text_model_2_conv1d_1_biasIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:Ј
AssignVariableOp_5AssignVariableOp/assignvariableop_5_text_model_2_conv1d_2_kernelIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:Ї
AssignVariableOp_6AssignVariableOp-assignvariableop_6_text_model_2_conv1d_2_biasIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:Ј
AssignVariableOp_7AssignVariableOp/assignvariableop_7_text_model_2_dense_48_kernelIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0Ї
AssignVariableOp_8AssignVariableOp-assignvariableop_8_text_model_2_dense_48_biasIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0Ј
AssignVariableOp_9AssignVariableOp/assignvariableop_9_text_model_2_dense_49_kernelIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:љ
AssignVariableOp_10AssignVariableOp.assignvariableop_10_text_model_2_dense_49_biasIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0	ѓ
AssignVariableOp_11AssignVariableOp assignvariableop_11_rmsprop_iterIdentity_11:output:0*
dtype0	*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:Ѓ
AssignVariableOp_12AssignVariableOp!assignvariableop_12_rmsprop_decayIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:І
AssignVariableOp_13AssignVariableOp)assignvariableop_13_rmsprop_learning_rateIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:є
AssignVariableOp_14AssignVariableOp$assignvariableop_14_rmsprop_momentumIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0Ђ
AssignVariableOp_15AssignVariableOpassignvariableop_15_rmsprop_rhoIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:{
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0{
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0д
AssignVariableOp_18AssignVariableOpDassignvariableop_18_rmsprop_text_model_2_embedding_29_embeddings_rmsIdentity_18:output:0*
_output_shapes
 *
dtype0P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:ю
AssignVariableOp_19AssignVariableOp:assignvariableop_19_rmsprop_text_model_2_conv1d_kernel_rmsIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:џ
AssignVariableOp_20AssignVariableOp8assignvariableop_20_rmsprop_text_model_2_conv1d_bias_rmsIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:ъ
AssignVariableOp_21AssignVariableOp<assignvariableop_21_rmsprop_text_model_2_conv1d_1_kernel_rmsIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0ю
AssignVariableOp_22AssignVariableOp:assignvariableop_22_rmsprop_text_model_2_conv1d_1_bias_rmsIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
_output_shapes
:*
T0ъ
AssignVariableOp_23AssignVariableOp<assignvariableop_23_rmsprop_text_model_2_conv1d_2_kernel_rmsIdentity_23:output:0*
_output_shapes
 *
dtype0P
Identity_24IdentityRestoreV2:tensors:24*
_output_shapes
:*
T0ю
AssignVariableOp_24AssignVariableOp:assignvariableop_24_rmsprop_text_model_2_conv1d_2_bias_rmsIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:ъ
AssignVariableOp_25AssignVariableOp<assignvariableop_25_rmsprop_text_model_2_dense_48_kernel_rmsIdentity_25:output:0*
_output_shapes
 *
dtype0P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:ю
AssignVariableOp_26AssignVariableOp:assignvariableop_26_rmsprop_text_model_2_dense_48_bias_rmsIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:ъ
AssignVariableOp_27AssignVariableOp<assignvariableop_27_rmsprop_text_model_2_dense_49_kernel_rmsIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:ю
AssignVariableOp_28AssignVariableOp:assignvariableop_28_rmsprop_text_model_2_dense_49_bias_rmsIdentity_28:output:0*
dtype0*
_output_shapes
 ї
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:х
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 ═
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ┌
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_30Identity_30:output:0*Ѕ
_input_shapesx
v: :::::::::::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : 
█
ф
)__inference_dense_48_layer_call_fn_582476

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_582023*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:         ђ*-
_gradient_op_typePartitionedCall-582029Ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*/
_input_shapes
:         г::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
Э.
Е
F__inference_text_model_layer_call_and_return_conditional_losses_582142
input_1/
+embedding_29_statefulpartitionedcall_args_1)
%conv1d_statefulpartitionedcall_args_1)
%conv1d_statefulpartitionedcall_args_2+
'conv1d_1_statefulpartitionedcall_args_1+
'conv1d_1_statefulpartitionedcall_args_2+
'conv1d_2_statefulpartitionedcall_args_1+
'conv1d_2_statefulpartitionedcall_args_2+
'dense_48_statefulpartitionedcall_args_1+
'dense_48_statefulpartitionedcall_args_2+
'dense_49_statefulpartitionedcall_args_1+
'dense_49_statefulpartitionedcall_args_2
identityѕбconv1d/StatefulPartitionedCallб conv1d_1/StatefulPartitionedCallб conv1d_2/StatefulPartitionedCallб dense_48/StatefulPartitionedCallб dense_49/StatefulPartitionedCallб$embedding_29/StatefulPartitionedCall№
$embedding_29/StatefulPartitionedCallStatefulPartitionedCallinput_1+embedding_29_statefulpartitionedcall_args_1*
Tin
2*,
_output_shapes
:         2╚*-
_gradient_op_typePartitionedCall-581989*Q
fLRJ
H__inference_embedding_29_layer_call_and_return_conditional_losses_581983*
Tout
2**
config_proto

CPU

GPU 2J 8ф
conv1d/StatefulPartitionedCallStatefulPartitionedCall-embedding_29/StatefulPartitionedCall:output:0%conv1d_statefulpartitionedcall_args_1%conv1d_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-581884*K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_581878*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:         1d▄
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         d*-
_gradient_op_typePartitionedCall-581964*Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958▓
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall-embedding_29/StatefulPartitionedCall:output:0'conv1d_1_statefulpartitionedcall_args_1'conv1d_1_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*+
_output_shapes
:         0d*
Tin
2*-
_gradient_op_typePartitionedCall-581914*M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_581908*
Tout
2Я
&global_max_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         d*
Tin
2*-
_gradient_op_typePartitionedCall-581964*Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958*
Tout
2▓
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall-embedding_29/StatefulPartitionedCall:output:0'conv1d_2_statefulpartitionedcall_args_1'conv1d_2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-581944*M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_581938*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:         /dЯ
&global_max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         d*
Tin
2*-
_gradient_op_typePartitionedCall-581964*Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958V
concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: ш
concatConcatV2-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d_1/PartitionedCall:output:0/global_max_pooling1d_2/PartitionedCall:output:0concat/axis:output:0*
T0*
N*(
_output_shapes
:         гЉ
 dense_48/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0'dense_48_statefulpartitionedcall_args_1'dense_48_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:         ђ*
Tin
2*-
_gradient_op_typePartitionedCall-582029*M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_582023*
Tout
2┼
dropout/PartitionedCallPartitionedCall)dense_48/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-582079*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_582067*
Tout
2**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:         ђ*
Tin
2А
 dense_49/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0'dense_49_statefulpartitionedcall_args_1'dense_49_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2*-
_gradient_op_typePartitionedCall-582101*M
fHRF
D__inference_dense_49_layer_call_and_return_conditional_losses_582095┼
IdentityIdentity)dense_49/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*R
_input_shapesA
?:         2:::::::::::2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2L
$embedding_29/StatefulPartitionedCall$embedding_29/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : 
░
b
C__inference_dropout_layer_call_and_return_conditional_losses_582496

inputs
identityѕQ
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *═╠L>C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:         ђї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Б
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*(
_output_shapes
:         ђ*
T0Ћ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:         ђR
dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0і
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:         ђb
dropout/mulMulinputsdropout/truediv:z:0*(
_output_shapes
:         ђ*
T0p
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:         ђj
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђZ
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
»Y
Н
F__inference_text_model_layer_call_and_return_conditional_losses_582410

inputs>
:embedding_29_embedding_lookup_read_readvariableop_resource6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource+
'dense_48_matmul_readvariableop_resource,
(dense_48_biasadd_readvariableop_resource+
'dense_49_matmul_readvariableop_resource,
(dense_49_biasadd_readvariableop_resource
identityѕбconv1d/BiasAdd/ReadVariableOpб)conv1d/conv1d/ExpandDims_1/ReadVariableOpбconv1d_1/BiasAdd/ReadVariableOpб+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpбconv1d_2/BiasAdd/ReadVariableOpб+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpбdense_48/BiasAdd/ReadVariableOpбdense_48/MatMul/ReadVariableOpбdense_49/BiasAdd/ReadVariableOpбdense_49/MatMul/ReadVariableOpбembedding_29/embedding_lookupб1embedding_29/embedding_lookup/Read/ReadVariableOpП
1embedding_29/embedding_lookup/Read/ReadVariableOpReadVariableOp:embedding_29_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*!
_output_shapes
:║Ь╚Ў
&embedding_29/embedding_lookup/IdentityIdentity9embedding_29/embedding_lookup/Read/ReadVariableOp:value:0*
T0*!
_output_shapes
:║Ь╚Т
embedding_29/embedding_lookupResourceGather:embedding_29_embedding_lookup_read_readvariableop_resourceinputs2^embedding_29/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*,
_output_shapes
:         2╚*D
_class:
86loc:@embedding_29/embedding_lookup/Read/ReadVariableOp*
Tindices0Є
(embedding_29/embedding_lookup/Identity_1Identity&embedding_29/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*,
_output_shapes
:         2╚*
T0*D
_class:
86loc:@embedding_29/embedding_lookup/Read/ReadVariableOpъ
(embedding_29/embedding_lookup/Identity_2Identity1embedding_29/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         2╚^
conv1d/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: ╗
conv1d/conv1d/ExpandDims
ExpandDims1embedding_29/embedding_lookup/Identity_2:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         2╚¤
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:╚d`
conv1d/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: Х
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:╚d┬
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
strides
*
paddingVALID*/
_output_shapes
:         1d*
T0Ё
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*+
_output_shapes
:         1d*
squeeze_dims
*
T0«
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dќ
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         1db
conv1d/ReluReluconv1d/BiasAdd:output:0*+
_output_shapes
:         1d*
T0l
*global_max_pooling1d/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: А
global_max_pooling1d/MaxMaxconv1d/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         d`
conv1d_1/conv1d/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :┐
conv1d_1/conv1d/ExpandDims
ExpandDims1embedding_29/embedding_lookup/Identity_2:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         2╚М
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:╚db
 conv1d_1/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: ╝
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:╚d╚
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:         0dЅ
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
squeeze_dims
*
T0*+
_output_shapes
:         0d▓
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dю
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         0df
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:         0dn
,global_max_pooling1d_1/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: Д
global_max_pooling1d_1/MaxMaxconv1d_1/Relu:activations:05global_max_pooling1d_1/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         d`
conv1d_2/conv1d/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :┐
conv1d_2/conv1d/ExpandDims
ExpandDims1embedding_29/embedding_lookup/Identity_2:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*0
_output_shapes
:         2╚*
T0М
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:╚db
 conv1d_2/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: ╝
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*'
_output_shapes
:╚d*
T0╚
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
paddingVALID*/
_output_shapes
:         /d*
T0*
strides
Ѕ
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*+
_output_shapes
:         /d*
squeeze_dims
*
T0▓
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dю
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*+
_output_shapes
:         /d*
T0f
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:         /dn
,global_max_pooling1d_2/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: Д
global_max_pooling1d_2/MaxMaxconv1d_2/Relu:activations:05global_max_pooling1d_2/Max/reduction_indices:output:0*'
_output_shapes
:         d*
T0V
concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: Л
concatConcatV2!global_max_pooling1d/Max:output:0#global_max_pooling1d_1/Max:output:0#global_max_pooling1d_2/Max:output:0concat/axis:output:0*(
_output_shapes
:         г*
T0*
NХ
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
гђЁ
dense_48/MatMulMatMulconcat:output:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ│
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђњ
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђc
dense_48/ReluReludense_48/BiasAdd:output:0*
T0*(
_output_shapes
:         ђl
dropout/IdentityIdentitydense_48/Relu:activations:0*
T0*(
_output_shapes
:         ђх
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђј
dense_49/MatMulMatMuldropout/Identity:output:0&dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ▓
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0Љ
dense_49/BiasAddBiasAdddense_49/MatMul:product:0'dense_49/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0h
dense_49/SoftmaxSoftmaxdense_49/BiasAdd:output:0*
T0*'
_output_shapes
:         е
IdentityIdentitydense_49/Softmax:softmax:0^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp ^dense_49/BiasAdd/ReadVariableOp^dense_49/MatMul/ReadVariableOp^embedding_29/embedding_lookup2^embedding_29/embedding_lookup/Read/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*R
_input_shapesA
?:         2:::::::::::2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2f
1embedding_29/embedding_lookup/Read/ReadVariableOp1embedding_29/embedding_lookup/Read/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2>
embedding_29/embedding_lookupembedding_29/embedding_lookup2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : 
Џ
і
-__inference_embedding_29_layer_call_fn_582458

inputs"
statefulpartitionedcall_args_1
identityѕбStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-581989*Q
fLRJ
H__inference_embedding_29_layer_call_and_return_conditional_losses_581983*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*,
_output_shapes
:         2╚Є
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         2╚"
identityIdentity:output:0**
_input_shapes
:         2:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: 
О	
П
D__inference_dense_49_layer_call_and_return_conditional_losses_582522

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         і
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*/
_input_shapes
:         ђ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Д
Ы
+__inference_text_model_layer_call_fn_582426

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identityѕбStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11*-
_gradient_op_typePartitionedCall-582173*O
fJRH
F__inference_text_model_layer_call_and_return_conditional_losses_582172*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*R
_input_shapesA
?:         2:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :	 :
 : :& "
 
_user_specified_nameinputs: : : : : 
ў0
╦
F__inference_text_model_layer_call_and_return_conditional_losses_582113
input_1/
+embedding_29_statefulpartitionedcall_args_1)
%conv1d_statefulpartitionedcall_args_1)
%conv1d_statefulpartitionedcall_args_2+
'conv1d_1_statefulpartitionedcall_args_1+
'conv1d_1_statefulpartitionedcall_args_2+
'conv1d_2_statefulpartitionedcall_args_1+
'conv1d_2_statefulpartitionedcall_args_2+
'dense_48_statefulpartitionedcall_args_1+
'dense_48_statefulpartitionedcall_args_2+
'dense_49_statefulpartitionedcall_args_1+
'dense_49_statefulpartitionedcall_args_2
identityѕбconv1d/StatefulPartitionedCallб conv1d_1/StatefulPartitionedCallб conv1d_2/StatefulPartitionedCallб dense_48/StatefulPartitionedCallб dense_49/StatefulPartitionedCallбdropout/StatefulPartitionedCallб$embedding_29/StatefulPartitionedCall№
$embedding_29/StatefulPartitionedCallStatefulPartitionedCallinput_1+embedding_29_statefulpartitionedcall_args_1*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*,
_output_shapes
:         2╚*-
_gradient_op_typePartitionedCall-581989*Q
fLRJ
H__inference_embedding_29_layer_call_and_return_conditional_losses_581983ф
conv1d/StatefulPartitionedCallStatefulPartitionedCall-embedding_29/StatefulPartitionedCall:output:0%conv1d_statefulpartitionedcall_args_1%conv1d_statefulpartitionedcall_args_2*+
_output_shapes
:         1d*
Tin
2*-
_gradient_op_typePartitionedCall-581884*K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_581878*
Tout
2**
config_proto

CPU

GPU 2J 8▄
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         d*
Tin
2*-
_gradient_op_typePartitionedCall-581964*Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958▓
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall-embedding_29/StatefulPartitionedCall:output:0'conv1d_1_statefulpartitionedcall_args_1'conv1d_1_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:         0d*-
_gradient_op_typePartitionedCall-581914*M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_581908*
Tout
2Я
&global_max_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-581964*Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         d*
Tin
2▓
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall-embedding_29/StatefulPartitionedCall:output:0'conv1d_2_statefulpartitionedcall_args_1'conv1d_2_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:         /d*-
_gradient_op_typePartitionedCall-581944*M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_581938*
Tout
2Я
&global_max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         d*
Tin
2*-
_gradient_op_typePartitionedCall-581964V
concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: ш
concatConcatV2-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d_1/PartitionedCall:output:0/global_max_pooling1d_2/PartitionedCall:output:0concat/axis:output:0*
T0*
N*(
_output_shapes
:         гЉ
 dense_48/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0'dense_48_statefulpartitionedcall_args_1'dense_48_statefulpartitionedcall_args_2*
Tin
2*(
_output_shapes
:         ђ*-
_gradient_op_typePartitionedCall-582029*M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_582023*
Tout
2**
config_proto

CPU

GPU 2J 8Н
dropout/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:         ђ*
Tin
2*-
_gradient_op_typePartitionedCall-582071*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_582060*
Tout
2Е
 dense_49/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0'dense_49_statefulpartitionedcall_args_1'dense_49_statefulpartitionedcall_args_2*M
fHRF
D__inference_dense_49_layer_call_and_return_conditional_losses_582095*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         *-
_gradient_op_typePartitionedCall-582101у
IdentityIdentity)dense_49/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall ^dropout/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*R
_input_shapesA
?:         2:::::::::::2L
$embedding_29/StatefulPartitionedCall$embedding_29/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : 
ш
Q
5__inference_global_max_pooling1d_layer_call_fn_581967

inputs
identity»
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-581964*Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958*
Tout
2**
config_proto

CPU

GPU 2J 8*0
_output_shapes
:                  *
Tin
2i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*<
_input_shapes+
):'                           :& "
 
_user_specified_nameinputs
ф
з
+__inference_text_model_layer_call_fn_582233
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         *-
_gradient_op_typePartitionedCall-582219*O
fJRH
F__inference_text_model_layer_call_and_return_conditional_losses_582218*
Tout
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*R
_input_shapesA
?:         2:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 
ђ
ф
)__inference_conv1d_2_layer_call_fn_581949

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*4
_output_shapes"
 :                  d*
Tin
2*-
_gradient_op_typePartitionedCall-581944*M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_581938*
Tout
2Ј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*4
_output_shapes"
 :                  d*
T0"
identityIdentity:output:0*<
_input_shapes+
):                  ╚::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
б
ш
B__inference_conv1d_layer_call_and_return_conditional_losses_581878

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpW
conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: І
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*9
_output_shapes'
%:#                  ╚*
T0┴
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:╚dY
conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: А
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*'
_output_shapes
:╚d*
T0Х
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*
strides
*
paddingVALID*8
_output_shapes&
$:"                  dђ
conv1d/SqueezeSqueezeconv1d:output:0*4
_output_shapes"
 :                  d*
squeeze_dims
*
T0а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dі
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  d]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :                  dЦ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*4
_output_shapes"
 :                  d*
T0"
identityIdentity:output:0*<
_input_shapes+
):                  ╚::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
п	
П
D__inference_dense_48_layer_call_and_return_conditional_losses_582023

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpц
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
гђj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђw
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*/
_input_shapes
:         г::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Ћ0
╩
F__inference_text_model_layer_call_and_return_conditional_losses_582172

inputs/
+embedding_29_statefulpartitionedcall_args_1)
%conv1d_statefulpartitionedcall_args_1)
%conv1d_statefulpartitionedcall_args_2+
'conv1d_1_statefulpartitionedcall_args_1+
'conv1d_1_statefulpartitionedcall_args_2+
'conv1d_2_statefulpartitionedcall_args_1+
'conv1d_2_statefulpartitionedcall_args_2+
'dense_48_statefulpartitionedcall_args_1+
'dense_48_statefulpartitionedcall_args_2+
'dense_49_statefulpartitionedcall_args_1+
'dense_49_statefulpartitionedcall_args_2
identityѕбconv1d/StatefulPartitionedCallб conv1d_1/StatefulPartitionedCallб conv1d_2/StatefulPartitionedCallб dense_48/StatefulPartitionedCallб dense_49/StatefulPartitionedCallбdropout/StatefulPartitionedCallб$embedding_29/StatefulPartitionedCallЬ
$embedding_29/StatefulPartitionedCallStatefulPartitionedCallinputs+embedding_29_statefulpartitionedcall_args_1*Q
fLRJ
H__inference_embedding_29_layer_call_and_return_conditional_losses_581983*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*,
_output_shapes
:         2╚*-
_gradient_op_typePartitionedCall-581989ф
conv1d/StatefulPartitionedCallStatefulPartitionedCall-embedding_29/StatefulPartitionedCall:output:0%conv1d_statefulpartitionedcall_args_1%conv1d_statefulpartitionedcall_args_2*+
_output_shapes
:         1d*
Tin
2*-
_gradient_op_typePartitionedCall-581884*K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_581878*
Tout
2**
config_proto

CPU

GPU 2J 8▄
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         d*-
_gradient_op_typePartitionedCall-581964▓
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall-embedding_29/StatefulPartitionedCall:output:0'conv1d_1_statefulpartitionedcall_args_1'conv1d_1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-581914*M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_581908*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:         0dЯ
&global_max_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-581964*Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         d▓
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall-embedding_29/StatefulPartitionedCall:output:0'conv1d_2_statefulpartitionedcall_args_1'conv1d_2_statefulpartitionedcall_args_2*M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_581938*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:         /d*-
_gradient_op_typePartitionedCall-581944Я
&global_max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         d*
Tin
2*-
_gradient_op_typePartitionedCall-581964*Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958V
concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: ш
concatConcatV2-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d_1/PartitionedCall:output:0/global_max_pooling1d_2/PartitionedCall:output:0concat/axis:output:0*
T0*
N*(
_output_shapes
:         гЉ
 dense_48/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0'dense_48_statefulpartitionedcall_args_1'dense_48_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:         ђ*-
_gradient_op_typePartitionedCall-582029*M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_582023Н
dropout/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:         ђ*
Tin
2*-
_gradient_op_typePartitionedCall-582071*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_582060*
Tout
2Е
 dense_49/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0'dense_49_statefulpartitionedcall_args_1'dense_49_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2*-
_gradient_op_typePartitionedCall-582101*M
fHRF
D__inference_dense_49_layer_call_and_return_conditional_losses_582095*
Tout
2у
IdentityIdentity)dense_49/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall ^dropout/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*R
_input_shapesA
?:         2:::::::::::2L
$embedding_29/StatefulPartitionedCall$embedding_29/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : 
■
В
$__inference_signature_wrapper_582255
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identityѕбStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2*-
_gradient_op_typePartitionedCall-582241**
f%R#
!__inference__wrapped_model_581859ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*R
_input_shapesA
?:         2:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :	 :
 : :' #
!
_user_specified_name	input_1: : : : 
Ч
е
'__inference_conv1d_layer_call_fn_581889

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*4
_output_shapes"
 :                  d*
Tin
2*-
_gradient_op_typePartitionedCall-581884*K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_581878Ј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  d"
identityIdentity:output:0*<
_input_shapes+
):                  ╚::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
ђ
ф
)__inference_conv1d_1_layer_call_fn_581919

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-581914*M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_581908*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*4
_output_shapes"
 :                  dЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  d"
identityIdentity:output:0*<
_input_shapes+
):                  ╚::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
ц
э
D__inference_conv1d_2_layer_call_and_return_conditional_losses_581938

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpW
conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: І
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  ╚┴
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:╚dY
conv1d/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : А
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*'
_output_shapes
:╚d*
T0Х
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*
strides
*
paddingVALID*8
_output_shapes&
$:"                  dђ
conv1d/SqueezeSqueezeconv1d:output:0*
squeeze_dims
*
T0*4
_output_shapes"
 :                  dа
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dі
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*4
_output_shapes"
 :                  d*
T0]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :                  dЦ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :                  d"
identityIdentity:output:0*<
_input_shapes+
):                  ╚::2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
О	
П
D__inference_dense_49_layer_call_and_return_conditional_losses_582095

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*'
_output_shapes
:         *
T0і
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*Ф
serving_defaultЌ
;
input_10
serving_default_input_1:0         2<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:ык
┴
	embedding

cnn_layer1

cnn_layer2

cnn_layer3
pool
dense_1
dropout

last_dense
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
*z&call_and_return_all_conditional_losses
{_default_save_signature
|__call__"Ѓ
_tf_keras_modelж{"class_name": "TEXT_MODEL", "name": "text_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "TEXT_MODEL"}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
у

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
*}&call_and_return_all_conditional_losses
~__call__"╚
_tf_keras_layer«{"class_name": "Embedding", "name": "embedding_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, null], "config": {"name": "embedding_29", "trainable": true, "batch_input_shape": [null, null], "dtype": "float32", "input_dim": 30522, "output_dim": 200, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}
с

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*&call_and_return_all_conditional_losses
ђ__call__"й
_tf_keras_layerБ{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 100, "kernel_size": [2], "strides": [1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 200}}}}
У

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+Ђ&call_and_return_all_conditional_losses
ѓ__call__"┴
_tf_keras_layerД{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 100, "kernel_size": [3], "strides": [1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 200}}}}
У

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
+Ѓ&call_and_return_all_conditional_losses
ё__call__"┴
_tf_keras_layerД{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 100, "kernel_size": [4], "strides": [1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 200}}}}
М
&regularization_losses
'	variables
(trainable_variables
)	keras_api
+Ё&call_and_return_all_conditional_losses
є__call__"┬
_tf_keras_layerе{"class_name": "GlobalMaxPooling1D", "name": "global_max_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "global_max_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
э

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
+Є&call_and_return_all_conditional_losses
ѕ__call__"л
_tf_keras_layerХ{"class_name": "Dense", "name": "dense_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_48", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}}}
Г
0regularization_losses
1	variables
2trainable_variables
3	keras_api
+Ѕ&call_and_return_all_conditional_losses
і__call__"ю
_tf_keras_layerѓ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
щ

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
+І&call_and_return_all_conditional_losses
ї__call__"м
_tf_keras_layerИ{"class_name": "Dense", "name": "dense_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_49", "trainable": true, "dtype": "float32", "units": 18, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
╦
:iter
	;decay
<learning_rate
=momentum
>rho	rmso	rmsp	rmsq	rmsr	rmss	 rmst	!rmsu	*rmsv	+rmsw	4rmsx	5rmsy"
	optimizer
 "
trackable_list_wrapper
n
0
1
2
3
4
 5
!6
*7
+8
49
510"
trackable_list_wrapper
n
0
1
2
3
4
 5
!6
*7
+8
49
510"
trackable_list_wrapper
и
?layer_regularization_losses

regularization_losses
@metrics
	variables
trainable_variables
Anon_trainable_variables

Blayers
|__call__
{_default_save_signature
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
-
Їserving_default"
signature_map
9:7║Ь╚2$text_model_2/embedding_29/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
џ
regularization_losses
Clayer_regularization_losses
Dmetrics
	variables
trainable_variables
Enon_trainable_variables

Flayers
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
1:/╚d2text_model_2/conv1d/kernel
&:$d2text_model_2/conv1d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Џ
regularization_losses
Glayer_regularization_losses
Hmetrics
	variables
trainable_variables
Inon_trainable_variables

Jlayers
ђ__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
3:1╚d2text_model_2/conv1d_1/kernel
(:&d2text_model_2/conv1d_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Ю
regularization_losses
Klayer_regularization_losses
Lmetrics
	variables
trainable_variables
Mnon_trainable_variables

Nlayers
ѓ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
3:1╚d2text_model_2/conv1d_2/kernel
(:&d2text_model_2/conv1d_2/bias
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
Ю
"regularization_losses
Olayer_regularization_losses
Pmetrics
#	variables
$trainable_variables
Qnon_trainable_variables

Rlayers
ё__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
&regularization_losses
Slayer_regularization_losses
Tmetrics
'	variables
(trainable_variables
Unon_trainable_variables

Vlayers
є__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
0:.
гђ2text_model_2/dense_48/kernel
):'ђ2text_model_2/dense_48/bias
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
Ю
,regularization_losses
Wlayer_regularization_losses
Xmetrics
-	variables
.trainable_variables
Ynon_trainable_variables

Zlayers
ѕ__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
0regularization_losses
[layer_regularization_losses
\metrics
1	variables
2trainable_variables
]non_trainable_variables

^layers
і__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
/:-	ђ2text_model_2/dense_49/kernel
(:&2text_model_2/dense_49/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
Ю
6regularization_losses
_layer_regularization_losses
`metrics
7	variables
8trainable_variables
anon_trainable_variables

blayers
ї__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
 "
trackable_list_wrapper
'
c0"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ю
	dtotal
	ecount
f
_fn_kwargs
gregularization_losses
h	variables
itrainable_variables
j	keras_api
+ј&call_and_return_all_conditional_losses
Ј__call__"т
_tf_keras_layer╦{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
gregularization_losses
klayer_regularization_losses
lmetrics
h	variables
itrainable_variables
mnon_trainable_variables

nlayers
Ј__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
C:A║Ь╚20RMSprop/text_model_2/embedding_29/embeddings/rms
;:9╚d2&RMSprop/text_model_2/conv1d/kernel/rms
0:.d2$RMSprop/text_model_2/conv1d/bias/rms
=:;╚d2(RMSprop/text_model_2/conv1d_1/kernel/rms
2:0d2&RMSprop/text_model_2/conv1d_1/bias/rms
=:;╚d2(RMSprop/text_model_2/conv1d_2/kernel/rms
2:0d2&RMSprop/text_model_2/conv1d_2/bias/rms
::8
гђ2(RMSprop/text_model_2/dense_48/kernel/rms
3:1ђ2&RMSprop/text_model_2/dense_48/bias/rms
9:7	ђ2(RMSprop/text_model_2/dense_49/kernel/rms
2:02&RMSprop/text_model_2/dense_49/bias/rms
о2М
F__inference_text_model_layer_call_and_return_conditional_losses_582341
F__inference_text_model_layer_call_and_return_conditional_losses_582410
F__inference_text_model_layer_call_and_return_conditional_losses_582113
F__inference_text_model_layer_call_and_return_conditional_losses_582142░
Д▓Б
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
▀2▄
!__inference__wrapped_model_581859Х
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *&б#
!і
input_1         2
Ж2у
+__inference_text_model_layer_call_fn_582233
+__inference_text_model_layer_call_fn_582442
+__inference_text_model_layer_call_fn_582426
+__inference_text_model_layer_call_fn_582187░
Д▓Б
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ы2№
H__inference_embedding_29_layer_call_and_return_conditional_losses_582452б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_embedding_29_layer_call_fn_582458б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ћ2њ
B__inference_conv1d_layer_call_and_return_conditional_losses_581878╦
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *+б(
&і#                  ╚
Щ2э
'__inference_conv1d_layer_call_fn_581889╦
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *+б(
&і#                  ╚
Ќ2ћ
D__inference_conv1d_1_layer_call_and_return_conditional_losses_581908╦
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *+б(
&і#                  ╚
Ч2щ
)__inference_conv1d_1_layer_call_fn_581919╦
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *+б(
&і#                  ╚
Ќ2ћ
D__inference_conv1d_2_layer_call_and_return_conditional_losses_581938╦
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *+б(
&і#                  ╚
Ч2щ
)__inference_conv1d_2_layer_call_fn_581949╦
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *+б(
&і#                  ╚
Ф2е
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958М
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *3б0
.і+'                           
љ2Ї
5__inference_global_max_pooling1d_layer_call_fn_581967М
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *3б0
.і+'                           
Ь2в
D__inference_dense_48_layer_call_and_return_conditional_losses_582469б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_48_layer_call_fn_582476б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
─2┴
C__inference_dropout_layer_call_and_return_conditional_losses_582496
C__inference_dropout_layer_call_and_return_conditional_losses_582501┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ј2І
(__inference_dropout_layer_call_fn_582506
(__inference_dropout_layer_call_fn_582511┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ь2в
D__inference_dense_49_layer_call_and_return_conditional_losses_582522б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_49_layer_call_fn_582529б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
3B1
$__inference_signature_wrapper_582255input_1
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 І
+__inference_text_model_layer_call_fn_582442\ !*+453б0
)б&
 і
inputs         2
p 
ф "і         Ћ
'__inference_conv1d_layer_call_fn_581889j=б:
3б0
.і+
inputs                  ╚
ф "%і"                  d│
F__inference_text_model_layer_call_and_return_conditional_losses_582341i !*+453б0
)б&
 і
inputs         2
p
ф "%б"
і
0         
џ ┤
F__inference_text_model_layer_call_and_return_conditional_losses_582113j !*+454б1
*б'
!і
input_1         2
p
ф "%б"
і
0         
џ ї
+__inference_text_model_layer_call_fn_582233] !*+454б1
*б'
!і
input_1         2
p 
ф "і         Б
5__inference_global_max_pooling1d_layer_call_fn_581967jEбB
;б8
6і3
inputs'                           
ф "!і                  Ц
C__inference_dropout_layer_call_and_return_conditional_losses_582501^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ ї
+__inference_text_model_layer_call_fn_582187] !*+454б1
*б'
!і
input_1         2
p
ф "і         ё
-__inference_embedding_29_layer_call_fn_582458S/б,
%б"
 і
inputs         2
ф "і         2╚│
F__inference_text_model_layer_call_and_return_conditional_losses_582410i !*+453б0
)б&
 і
inputs         2
p 
ф "%б"
і
0         
џ }
(__inference_dropout_layer_call_fn_582511Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђ}
(__inference_dropout_layer_call_fn_582506Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђ╦
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_581958wEбB
;б8
6і3
inputs'                           
ф ".б+
$і!
0                  
џ Ц
D__inference_dense_49_layer_call_and_return_conditional_losses_582522]450б-
&б#
!і
inputs         ђ
ф "%б"
і
0         
џ ┤
F__inference_text_model_layer_call_and_return_conditional_losses_582142j !*+454б1
*б'
!і
input_1         2
p 
ф "%б"
і
0         
џ й
B__inference_conv1d_layer_call_and_return_conditional_losses_581878w=б:
3б0
.і+
inputs                  ╚
ф "2б/
(і%
0                  d
џ д
D__inference_dense_48_layer_call_and_return_conditional_losses_582469^*+0б-
&б#
!і
inputs         г
ф "&б#
і
0         ђ
џ г
H__inference_embedding_29_layer_call_and_return_conditional_losses_582452`/б,
%б"
 і
inputs         2
ф "*б'
 і
0         2╚
џ Ќ
)__inference_conv1d_1_layer_call_fn_581919j=б:
3б0
.і+
inputs                  ╚
ф "%і"                  d┐
D__inference_conv1d_1_layer_call_and_return_conditional_losses_581908w=б:
3б0
.і+
inputs                  ╚
ф "2б/
(і%
0                  d
џ Д
$__inference_signature_wrapper_582255 !*+45;б8
б 
1ф.
,
input_1!і
input_1         2"3ф0
.
output_1"і
output_1         Ќ
)__inference_conv1d_2_layer_call_fn_581949j !=б:
3б0
.і+
inputs                  ╚
ф "%і"                  dЦ
C__inference_dropout_layer_call_and_return_conditional_losses_582496^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ ┐
D__inference_conv1d_2_layer_call_and_return_conditional_losses_581938w !=б:
3б0
.і+
inputs                  ╚
ф "2б/
(і%
0                  d
џ }
)__inference_dense_49_layer_call_fn_582529P450б-
&б#
!і
inputs         ђ
ф "і         Ў
!__inference__wrapped_model_581859t !*+450б-
&б#
!і
input_1         2
ф "3ф0
.
output_1"і
output_1         І
+__inference_text_model_layer_call_fn_582426\ !*+453б0
)б&
 і
inputs         2
p
ф "і         ~
)__inference_dense_48_layer_call_fn_582476Q*+0б-
&б#
!і
inputs         г
ф "і         ђ
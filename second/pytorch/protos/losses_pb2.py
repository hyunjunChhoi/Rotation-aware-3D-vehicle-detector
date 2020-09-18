# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: second/protos/losses.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='second/protos/losses.proto',
  package='second.protos',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x1asecond/protos/losses.proto\x12\rsecond.protos\"\xfb\x01\n\x04Loss\x12:\n\x11localization_loss\x18\x01 \x01(\x0b\x32\x1f.second.protos.LocalizationLoss\x12>\n\x13\x63lassification_loss\x18\x02 \x01(\x0b\x32!.second.protos.ClassificationLoss\x12;\n\x12hard_example_miner\x18\x03 \x01(\x0b\x32\x1f.second.protos.HardExampleMiner\x12\x1d\n\x15\x63lassification_weight\x18\x04 \x01(\x02\x12\x1b\n\x13localization_weight\x18\x05 \x01(\x02\"\x9d\x02\n\x10LocalizationLoss\x12@\n\x0bweighted_l2\x18\x01 \x01(\x0b\x32).second.protos.WeightedL2LocalizationLossH\x00\x12M\n\x12weighted_smooth_l1\x18\x02 \x01(\x0b\x32/.second.protos.WeightedSmoothL1LocalizationLossH\x00\x12\x42\n\x0cweighted_ghm\x18\x03 \x01(\x0b\x32*.second.protos.WeightedGHMLocalizationLossH\x00\x12\x1f\n\x17\x65ncode_rad_error_by_sin\x18\x04 \x01(\x08\x42\x13\n\x11localization_loss\"L\n\x1aWeightedL2LocalizationLoss\x12\x19\n\x11\x61nchorwise_output\x18\x01 \x01(\x08\x12\x13\n\x0b\x63ode_weight\x18\x02 \x03(\x02\"a\n WeightedSmoothL1LocalizationLoss\x12\x19\n\x11\x61nchorwise_output\x18\x01 \x01(\x08\x12\r\n\x05sigma\x18\x02 \x01(\x02\x12\x13\n\x0b\x63ode_weight\x18\x03 \x03(\x02\"y\n\x1bWeightedGHMLocalizationLoss\x12\x19\n\x11\x61nchorwise_output\x18\x01 \x01(\x08\x12\n\n\x02mu\x18\x02 \x01(\x02\x12\x0c\n\x04\x62ins\x18\x03 \x01(\x05\x12\x10\n\x08momentum\x18\x04 \x01(\x02\x12\x13\n\x0b\x63ode_weight\x18\x05 \x03(\x02\"\xfd\x03\n\x12\x43lassificationLoss\x12L\n\x10weighted_sigmoid\x18\x01 \x01(\x0b\x32\x30.second.protos.WeightedSigmoidClassificationLossH\x00\x12L\n\x10weighted_softmax\x18\x02 \x01(\x0b\x32\x30.second.protos.WeightedSoftmaxClassificationLossH\x00\x12T\n\x14\x62ootstrapped_sigmoid\x18\x03 \x01(\x0b\x32\x34.second.protos.BootstrappedSigmoidClassificationLossH\x00\x12O\n\x16weighted_sigmoid_focal\x18\x04 \x01(\x0b\x32-.second.protos.SigmoidFocalClassificationLossH\x00\x12O\n\x16weighted_softmax_focal\x18\x05 \x01(\x0b\x32-.second.protos.SoftmaxFocalClassificationLossH\x00\x12<\n\x0cweighted_ghm\x18\x06 \x01(\x0b\x32$.second.protos.GHMClassificationLossH\x00\x42\x15\n\x13\x63lassification_loss\">\n!WeightedSigmoidClassificationLoss\x12\x19\n\x11\x61nchorwise_output\x18\x01 \x01(\x08\"Y\n\x1eSigmoidFocalClassificationLoss\x12\x19\n\x11\x61nchorwise_output\x18\x01 \x01(\x08\x12\r\n\x05gamma\x18\x02 \x01(\x02\x12\r\n\x05\x61lpha\x18\x03 \x01(\x02\"Y\n\x1eSoftmaxFocalClassificationLoss\x12\x19\n\x11\x61nchorwise_output\x18\x01 \x01(\x08\x12\r\n\x05gamma\x18\x02 \x01(\x02\x12\r\n\x05\x61lpha\x18\x03 \x01(\x02\"R\n\x15GHMClassificationLoss\x12\x19\n\x11\x61nchorwise_output\x18\x01 \x01(\x08\x12\x0c\n\x04\x62ins\x18\x02 \x01(\x05\x12\x10\n\x08momentum\x18\x03 \x01(\x02\"S\n!WeightedSoftmaxClassificationLoss\x12\x19\n\x11\x61nchorwise_output\x18\x01 \x01(\x08\x12\x13\n\x0blogit_scale\x18\x02 \x01(\x02\"i\n%BootstrappedSigmoidClassificationLoss\x12\r\n\x05\x61lpha\x18\x01 \x01(\x02\x12\x16\n\x0ehard_bootstrap\x18\x02 \x01(\x08\x12\x19\n\x11\x61nchorwise_output\x18\x03 \x01(\x08\"\x82\x02\n\x10HardExampleMiner\x12\x19\n\x11num_hard_examples\x18\x01 \x01(\x05\x12\x15\n\riou_threshold\x18\x02 \x01(\x02\x12;\n\tloss_type\x18\x03 \x01(\x0e\x32(.second.protos.HardExampleMiner.LossType\x12\"\n\x1amax_negatives_per_positive\x18\x04 \x01(\x05\x12\x1f\n\x17min_negatives_per_image\x18\x05 \x01(\x05\":\n\x08LossType\x12\x08\n\x04\x42OTH\x10\x00\x12\x12\n\x0e\x43LASSIFICATION\x10\x01\x12\x10\n\x0cLOCALIZATION\x10\x02\x62\x06proto3')
)



_HARDEXAMPLEMINER_LOSSTYPE = _descriptor.EnumDescriptor(
  name='LossType',
  full_name='second.protos.HardExampleMiner.LossType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='BOTH', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CLASSIFICATION', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOCALIZATION', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=2122,
  serialized_end=2180,
)
_sym_db.RegisterEnumDescriptor(_HARDEXAMPLEMINER_LOSSTYPE)


_LOSS = _descriptor.Descriptor(
  name='Loss',
  full_name='second.protos.Loss',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='localization_loss', full_name='second.protos.Loss.localization_loss', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='classification_loss', full_name='second.protos.Loss.classification_loss', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hard_example_miner', full_name='second.protos.Loss.hard_example_miner', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='classification_weight', full_name='second.protos.Loss.classification_weight', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='localization_weight', full_name='second.protos.Loss.localization_weight', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=46,
  serialized_end=297,
)


_LOCALIZATIONLOSS = _descriptor.Descriptor(
  name='LocalizationLoss',
  full_name='second.protos.LocalizationLoss',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='weighted_l2', full_name='second.protos.LocalizationLoss.weighted_l2', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weighted_smooth_l1', full_name='second.protos.LocalizationLoss.weighted_smooth_l1', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weighted_ghm', full_name='second.protos.LocalizationLoss.weighted_ghm', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='encode_rad_error_by_sin', full_name='second.protos.LocalizationLoss.encode_rad_error_by_sin', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='localization_loss', full_name='second.protos.LocalizationLoss.localization_loss',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=300,
  serialized_end=585,
)


_WEIGHTEDL2LOCALIZATIONLOSS = _descriptor.Descriptor(
  name='WeightedL2LocalizationLoss',
  full_name='second.protos.WeightedL2LocalizationLoss',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='anchorwise_output', full_name='second.protos.WeightedL2LocalizationLoss.anchorwise_output', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='code_weight', full_name='second.protos.WeightedL2LocalizationLoss.code_weight', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=587,
  serialized_end=663,
)


_WEIGHTEDSMOOTHL1LOCALIZATIONLOSS = _descriptor.Descriptor(
  name='WeightedSmoothL1LocalizationLoss',
  full_name='second.protos.WeightedSmoothL1LocalizationLoss',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='anchorwise_output', full_name='second.protos.WeightedSmoothL1LocalizationLoss.anchorwise_output', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sigma', full_name='second.protos.WeightedSmoothL1LocalizationLoss.sigma', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='code_weight', full_name='second.protos.WeightedSmoothL1LocalizationLoss.code_weight', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=665,
  serialized_end=762,
)


_WEIGHTEDGHMLOCALIZATIONLOSS = _descriptor.Descriptor(
  name='WeightedGHMLocalizationLoss',
  full_name='second.protos.WeightedGHMLocalizationLoss',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='anchorwise_output', full_name='second.protos.WeightedGHMLocalizationLoss.anchorwise_output', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mu', full_name='second.protos.WeightedGHMLocalizationLoss.mu', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bins', full_name='second.protos.WeightedGHMLocalizationLoss.bins', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='momentum', full_name='second.protos.WeightedGHMLocalizationLoss.momentum', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='code_weight', full_name='second.protos.WeightedGHMLocalizationLoss.code_weight', index=4,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=764,
  serialized_end=885,
)


_CLASSIFICATIONLOSS = _descriptor.Descriptor(
  name='ClassificationLoss',
  full_name='second.protos.ClassificationLoss',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='weighted_sigmoid', full_name='second.protos.ClassificationLoss.weighted_sigmoid', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weighted_softmax', full_name='second.protos.ClassificationLoss.weighted_softmax', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bootstrapped_sigmoid', full_name='second.protos.ClassificationLoss.bootstrapped_sigmoid', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weighted_sigmoid_focal', full_name='second.protos.ClassificationLoss.weighted_sigmoid_focal', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weighted_softmax_focal', full_name='second.protos.ClassificationLoss.weighted_softmax_focal', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weighted_ghm', full_name='second.protos.ClassificationLoss.weighted_ghm', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='classification_loss', full_name='second.protos.ClassificationLoss.classification_loss',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=888,
  serialized_end=1397,
)


_WEIGHTEDSIGMOIDCLASSIFICATIONLOSS = _descriptor.Descriptor(
  name='WeightedSigmoidClassificationLoss',
  full_name='second.protos.WeightedSigmoidClassificationLoss',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='anchorwise_output', full_name='second.protos.WeightedSigmoidClassificationLoss.anchorwise_output', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1399,
  serialized_end=1461,
)


_SIGMOIDFOCALCLASSIFICATIONLOSS = _descriptor.Descriptor(
  name='SigmoidFocalClassificationLoss',
  full_name='second.protos.SigmoidFocalClassificationLoss',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='anchorwise_output', full_name='second.protos.SigmoidFocalClassificationLoss.anchorwise_output', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gamma', full_name='second.protos.SigmoidFocalClassificationLoss.gamma', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='alpha', full_name='second.protos.SigmoidFocalClassificationLoss.alpha', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1463,
  serialized_end=1552,
)


_SOFTMAXFOCALCLASSIFICATIONLOSS = _descriptor.Descriptor(
  name='SoftmaxFocalClassificationLoss',
  full_name='second.protos.SoftmaxFocalClassificationLoss',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='anchorwise_output', full_name='second.protos.SoftmaxFocalClassificationLoss.anchorwise_output', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gamma', full_name='second.protos.SoftmaxFocalClassificationLoss.gamma', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='alpha', full_name='second.protos.SoftmaxFocalClassificationLoss.alpha', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1554,
  serialized_end=1643,
)


_GHMCLASSIFICATIONLOSS = _descriptor.Descriptor(
  name='GHMClassificationLoss',
  full_name='second.protos.GHMClassificationLoss',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='anchorwise_output', full_name='second.protos.GHMClassificationLoss.anchorwise_output', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bins', full_name='second.protos.GHMClassificationLoss.bins', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='momentum', full_name='second.protos.GHMClassificationLoss.momentum', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1645,
  serialized_end=1727,
)


_WEIGHTEDSOFTMAXCLASSIFICATIONLOSS = _descriptor.Descriptor(
  name='WeightedSoftmaxClassificationLoss',
  full_name='second.protos.WeightedSoftmaxClassificationLoss',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='anchorwise_output', full_name='second.protos.WeightedSoftmaxClassificationLoss.anchorwise_output', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='logit_scale', full_name='second.protos.WeightedSoftmaxClassificationLoss.logit_scale', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1729,
  serialized_end=1812,
)


_BOOTSTRAPPEDSIGMOIDCLASSIFICATIONLOSS = _descriptor.Descriptor(
  name='BootstrappedSigmoidClassificationLoss',
  full_name='second.protos.BootstrappedSigmoidClassificationLoss',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='alpha', full_name='second.protos.BootstrappedSigmoidClassificationLoss.alpha', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hard_bootstrap', full_name='second.protos.BootstrappedSigmoidClassificationLoss.hard_bootstrap', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='anchorwise_output', full_name='second.protos.BootstrappedSigmoidClassificationLoss.anchorwise_output', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1814,
  serialized_end=1919,
)


_HARDEXAMPLEMINER = _descriptor.Descriptor(
  name='HardExampleMiner',
  full_name='second.protos.HardExampleMiner',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_hard_examples', full_name='second.protos.HardExampleMiner.num_hard_examples', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='iou_threshold', full_name='second.protos.HardExampleMiner.iou_threshold', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loss_type', full_name='second.protos.HardExampleMiner.loss_type', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_negatives_per_positive', full_name='second.protos.HardExampleMiner.max_negatives_per_positive', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_negatives_per_image', full_name='second.protos.HardExampleMiner.min_negatives_per_image', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _HARDEXAMPLEMINER_LOSSTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1922,
  serialized_end=2180,
)

_LOSS.fields_by_name['localization_loss'].message_type = _LOCALIZATIONLOSS
_LOSS.fields_by_name['classification_loss'].message_type = _CLASSIFICATIONLOSS
_LOSS.fields_by_name['hard_example_miner'].message_type = _HARDEXAMPLEMINER
_LOCALIZATIONLOSS.fields_by_name['weighted_l2'].message_type = _WEIGHTEDL2LOCALIZATIONLOSS
_LOCALIZATIONLOSS.fields_by_name['weighted_smooth_l1'].message_type = _WEIGHTEDSMOOTHL1LOCALIZATIONLOSS
_LOCALIZATIONLOSS.fields_by_name['weighted_ghm'].message_type = _WEIGHTEDGHMLOCALIZATIONLOSS
_LOCALIZATIONLOSS.oneofs_by_name['localization_loss'].fields.append(
  _LOCALIZATIONLOSS.fields_by_name['weighted_l2'])
_LOCALIZATIONLOSS.fields_by_name['weighted_l2'].containing_oneof = _LOCALIZATIONLOSS.oneofs_by_name['localization_loss']
_LOCALIZATIONLOSS.oneofs_by_name['localization_loss'].fields.append(
  _LOCALIZATIONLOSS.fields_by_name['weighted_smooth_l1'])
_LOCALIZATIONLOSS.fields_by_name['weighted_smooth_l1'].containing_oneof = _LOCALIZATIONLOSS.oneofs_by_name['localization_loss']
_LOCALIZATIONLOSS.oneofs_by_name['localization_loss'].fields.append(
  _LOCALIZATIONLOSS.fields_by_name['weighted_ghm'])
_LOCALIZATIONLOSS.fields_by_name['weighted_ghm'].containing_oneof = _LOCALIZATIONLOSS.oneofs_by_name['localization_loss']
_CLASSIFICATIONLOSS.fields_by_name['weighted_sigmoid'].message_type = _WEIGHTEDSIGMOIDCLASSIFICATIONLOSS
_CLASSIFICATIONLOSS.fields_by_name['weighted_softmax'].message_type = _WEIGHTEDSOFTMAXCLASSIFICATIONLOSS
_CLASSIFICATIONLOSS.fields_by_name['bootstrapped_sigmoid'].message_type = _BOOTSTRAPPEDSIGMOIDCLASSIFICATIONLOSS
_CLASSIFICATIONLOSS.fields_by_name['weighted_sigmoid_focal'].message_type = _SIGMOIDFOCALCLASSIFICATIONLOSS
_CLASSIFICATIONLOSS.fields_by_name['weighted_softmax_focal'].message_type = _SOFTMAXFOCALCLASSIFICATIONLOSS
_CLASSIFICATIONLOSS.fields_by_name['weighted_ghm'].message_type = _GHMCLASSIFICATIONLOSS
_CLASSIFICATIONLOSS.oneofs_by_name['classification_loss'].fields.append(
  _CLASSIFICATIONLOSS.fields_by_name['weighted_sigmoid'])
_CLASSIFICATIONLOSS.fields_by_name['weighted_sigmoid'].containing_oneof = _CLASSIFICATIONLOSS.oneofs_by_name['classification_loss']
_CLASSIFICATIONLOSS.oneofs_by_name['classification_loss'].fields.append(
  _CLASSIFICATIONLOSS.fields_by_name['weighted_softmax'])
_CLASSIFICATIONLOSS.fields_by_name['weighted_softmax'].containing_oneof = _CLASSIFICATIONLOSS.oneofs_by_name['classification_loss']
_CLASSIFICATIONLOSS.oneofs_by_name['classification_loss'].fields.append(
  _CLASSIFICATIONLOSS.fields_by_name['bootstrapped_sigmoid'])
_CLASSIFICATIONLOSS.fields_by_name['bootstrapped_sigmoid'].containing_oneof = _CLASSIFICATIONLOSS.oneofs_by_name['classification_loss']
_CLASSIFICATIONLOSS.oneofs_by_name['classification_loss'].fields.append(
  _CLASSIFICATIONLOSS.fields_by_name['weighted_sigmoid_focal'])
_CLASSIFICATIONLOSS.fields_by_name['weighted_sigmoid_focal'].containing_oneof = _CLASSIFICATIONLOSS.oneofs_by_name['classification_loss']
_CLASSIFICATIONLOSS.oneofs_by_name['classification_loss'].fields.append(
  _CLASSIFICATIONLOSS.fields_by_name['weighted_softmax_focal'])
_CLASSIFICATIONLOSS.fields_by_name['weighted_softmax_focal'].containing_oneof = _CLASSIFICATIONLOSS.oneofs_by_name['classification_loss']
_CLASSIFICATIONLOSS.oneofs_by_name['classification_loss'].fields.append(
  _CLASSIFICATIONLOSS.fields_by_name['weighted_ghm'])
_CLASSIFICATIONLOSS.fields_by_name['weighted_ghm'].containing_oneof = _CLASSIFICATIONLOSS.oneofs_by_name['classification_loss']
_HARDEXAMPLEMINER.fields_by_name['loss_type'].enum_type = _HARDEXAMPLEMINER_LOSSTYPE
_HARDEXAMPLEMINER_LOSSTYPE.containing_type = _HARDEXAMPLEMINER
DESCRIPTOR.message_types_by_name['Loss'] = _LOSS
DESCRIPTOR.message_types_by_name['LocalizationLoss'] = _LOCALIZATIONLOSS
DESCRIPTOR.message_types_by_name['WeightedL2LocalizationLoss'] = _WEIGHTEDL2LOCALIZATIONLOSS
DESCRIPTOR.message_types_by_name['WeightedSmoothL1LocalizationLoss'] = _WEIGHTEDSMOOTHL1LOCALIZATIONLOSS
DESCRIPTOR.message_types_by_name['WeightedGHMLocalizationLoss'] = _WEIGHTEDGHMLOCALIZATIONLOSS
DESCRIPTOR.message_types_by_name['ClassificationLoss'] = _CLASSIFICATIONLOSS
DESCRIPTOR.message_types_by_name['WeightedSigmoidClassificationLoss'] = _WEIGHTEDSIGMOIDCLASSIFICATIONLOSS
DESCRIPTOR.message_types_by_name['SigmoidFocalClassificationLoss'] = _SIGMOIDFOCALCLASSIFICATIONLOSS
DESCRIPTOR.message_types_by_name['SoftmaxFocalClassificationLoss'] = _SOFTMAXFOCALCLASSIFICATIONLOSS
DESCRIPTOR.message_types_by_name['GHMClassificationLoss'] = _GHMCLASSIFICATIONLOSS
DESCRIPTOR.message_types_by_name['WeightedSoftmaxClassificationLoss'] = _WEIGHTEDSOFTMAXCLASSIFICATIONLOSS
DESCRIPTOR.message_types_by_name['BootstrappedSigmoidClassificationLoss'] = _BOOTSTRAPPEDSIGMOIDCLASSIFICATIONLOSS
DESCRIPTOR.message_types_by_name['HardExampleMiner'] = _HARDEXAMPLEMINER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Loss = _reflection.GeneratedProtocolMessageType('Loss', (_message.Message,), {
  'DESCRIPTOR' : _LOSS,
  '__module__' : 'second.protos.losses_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.Loss)
  })
_sym_db.RegisterMessage(Loss)

LocalizationLoss = _reflection.GeneratedProtocolMessageType('LocalizationLoss', (_message.Message,), {
  'DESCRIPTOR' : _LOCALIZATIONLOSS,
  '__module__' : 'second.protos.losses_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.LocalizationLoss)
  })
_sym_db.RegisterMessage(LocalizationLoss)

WeightedL2LocalizationLoss = _reflection.GeneratedProtocolMessageType('WeightedL2LocalizationLoss', (_message.Message,), {
  'DESCRIPTOR' : _WEIGHTEDL2LOCALIZATIONLOSS,
  '__module__' : 'second.protos.losses_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.WeightedL2LocalizationLoss)
  })
_sym_db.RegisterMessage(WeightedL2LocalizationLoss)

WeightedSmoothL1LocalizationLoss = _reflection.GeneratedProtocolMessageType('WeightedSmoothL1LocalizationLoss', (_message.Message,), {
  'DESCRIPTOR' : _WEIGHTEDSMOOTHL1LOCALIZATIONLOSS,
  '__module__' : 'second.protos.losses_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.WeightedSmoothL1LocalizationLoss)
  })
_sym_db.RegisterMessage(WeightedSmoothL1LocalizationLoss)

WeightedGHMLocalizationLoss = _reflection.GeneratedProtocolMessageType('WeightedGHMLocalizationLoss', (_message.Message,), {
  'DESCRIPTOR' : _WEIGHTEDGHMLOCALIZATIONLOSS,
  '__module__' : 'second.protos.losses_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.WeightedGHMLocalizationLoss)
  })
_sym_db.RegisterMessage(WeightedGHMLocalizationLoss)

ClassificationLoss = _reflection.GeneratedProtocolMessageType('ClassificationLoss', (_message.Message,), {
  'DESCRIPTOR' : _CLASSIFICATIONLOSS,
  '__module__' : 'second.protos.losses_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.ClassificationLoss)
  })
_sym_db.RegisterMessage(ClassificationLoss)

WeightedSigmoidClassificationLoss = _reflection.GeneratedProtocolMessageType('WeightedSigmoidClassificationLoss', (_message.Message,), {
  'DESCRIPTOR' : _WEIGHTEDSIGMOIDCLASSIFICATIONLOSS,
  '__module__' : 'second.protos.losses_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.WeightedSigmoidClassificationLoss)
  })
_sym_db.RegisterMessage(WeightedSigmoidClassificationLoss)

SigmoidFocalClassificationLoss = _reflection.GeneratedProtocolMessageType('SigmoidFocalClassificationLoss', (_message.Message,), {
  'DESCRIPTOR' : _SIGMOIDFOCALCLASSIFICATIONLOSS,
  '__module__' : 'second.protos.losses_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.SigmoidFocalClassificationLoss)
  })
_sym_db.RegisterMessage(SigmoidFocalClassificationLoss)

SoftmaxFocalClassificationLoss = _reflection.GeneratedProtocolMessageType('SoftmaxFocalClassificationLoss', (_message.Message,), {
  'DESCRIPTOR' : _SOFTMAXFOCALCLASSIFICATIONLOSS,
  '__module__' : 'second.protos.losses_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.SoftmaxFocalClassificationLoss)
  })
_sym_db.RegisterMessage(SoftmaxFocalClassificationLoss)

GHMClassificationLoss = _reflection.GeneratedProtocolMessageType('GHMClassificationLoss', (_message.Message,), {
  'DESCRIPTOR' : _GHMCLASSIFICATIONLOSS,
  '__module__' : 'second.protos.losses_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.GHMClassificationLoss)
  })
_sym_db.RegisterMessage(GHMClassificationLoss)

WeightedSoftmaxClassificationLoss = _reflection.GeneratedProtocolMessageType('WeightedSoftmaxClassificationLoss', (_message.Message,), {
  'DESCRIPTOR' : _WEIGHTEDSOFTMAXCLASSIFICATIONLOSS,
  '__module__' : 'second.protos.losses_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.WeightedSoftmaxClassificationLoss)
  })
_sym_db.RegisterMessage(WeightedSoftmaxClassificationLoss)

BootstrappedSigmoidClassificationLoss = _reflection.GeneratedProtocolMessageType('BootstrappedSigmoidClassificationLoss', (_message.Message,), {
  'DESCRIPTOR' : _BOOTSTRAPPEDSIGMOIDCLASSIFICATIONLOSS,
  '__module__' : 'second.protos.losses_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.BootstrappedSigmoidClassificationLoss)
  })
_sym_db.RegisterMessage(BootstrappedSigmoidClassificationLoss)

HardExampleMiner = _reflection.GeneratedProtocolMessageType('HardExampleMiner', (_message.Message,), {
  'DESCRIPTOR' : _HARDEXAMPLEMINER,
  '__module__' : 'second.protos.losses_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.HardExampleMiner)
  })
_sym_db.RegisterMessage(HardExampleMiner)


# @@protoc_insertion_point(module_scope)

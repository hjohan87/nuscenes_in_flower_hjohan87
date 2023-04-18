# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: flwr/proto/task.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from flwr.proto import node_pb2 as flwr_dot_proto_dot_node__pb2
from flwr.proto import transport_pb2 as flwr_dot_proto_dot_transport__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='flwr/proto/task.proto',
  package='flwr.proto',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x15\x66lwr/proto/task.proto\x12\nflwr.proto\x1a\x15\x66lwr/proto/node.proto\x1a\x1a\x66lwr/proto/transport.proto\"\x93\x02\n\x04Task\x12\"\n\x08producer\x18\x01 \x01(\x0b\x32\x10.flwr.proto.Node\x12\"\n\x08\x63onsumer\x18\x02 \x01(\x0b\x32\x10.flwr.proto.Node\x12\x12\n\ncreated_at\x18\x03 \x01(\t\x12\x14\n\x0c\x64\x65livered_at\x18\x04 \x01(\t\x12\x0b\n\x03ttl\x18\x05 \x01(\t\x12\x10\n\x08\x61ncestry\x18\x06 \x03(\t\x12<\n\x15legacy_server_message\x18\x65 \x01(\x0b\x32\x19.flwr.proto.ServerMessageB\x02\x18\x01\x12<\n\x15legacy_client_message\x18\x66 \x01(\x0b\x32\x19.flwr.proto.ClientMessageB\x02\x18\x01\"a\n\x07TaskIns\x12\x0f\n\x07task_id\x18\x01 \x01(\t\x12\x10\n\x08group_id\x18\x02 \x01(\t\x12\x13\n\x0bworkload_id\x18\x03 \x01(\t\x12\x1e\n\x04task\x18\x04 \x01(\x0b\x32\x10.flwr.proto.Task\"a\n\x07TaskRes\x12\x0f\n\x07task_id\x18\x01 \x01(\t\x12\x10\n\x08group_id\x18\x02 \x01(\t\x12\x13\n\x0bworkload_id\x18\x03 \x01(\t\x12\x1e\n\x04task\x18\x04 \x01(\x0b\x32\x10.flwr.proto.Taskb\x06proto3'
  ,
  dependencies=[flwr_dot_proto_dot_node__pb2.DESCRIPTOR,flwr_dot_proto_dot_transport__pb2.DESCRIPTOR,])




_TASK = _descriptor.Descriptor(
  name='Task',
  full_name='flwr.proto.Task',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='producer', full_name='flwr.proto.Task.producer', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='consumer', full_name='flwr.proto.Task.consumer', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='created_at', full_name='flwr.proto.Task.created_at', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='delivered_at', full_name='flwr.proto.Task.delivered_at', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ttl', full_name='flwr.proto.Task.ttl', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ancestry', full_name='flwr.proto.Task.ancestry', index=5,
      number=6, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='legacy_server_message', full_name='flwr.proto.Task.legacy_server_message', index=6,
      number=101, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\030\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='legacy_client_message', full_name='flwr.proto.Task.legacy_client_message', index=7,
      number=102, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\030\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=89,
  serialized_end=364,
)


_TASKINS = _descriptor.Descriptor(
  name='TaskIns',
  full_name='flwr.proto.TaskIns',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='task_id', full_name='flwr.proto.TaskIns.task_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='group_id', full_name='flwr.proto.TaskIns.group_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='workload_id', full_name='flwr.proto.TaskIns.workload_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='task', full_name='flwr.proto.TaskIns.task', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=366,
  serialized_end=463,
)


_TASKRES = _descriptor.Descriptor(
  name='TaskRes',
  full_name='flwr.proto.TaskRes',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='task_id', full_name='flwr.proto.TaskRes.task_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='group_id', full_name='flwr.proto.TaskRes.group_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='workload_id', full_name='flwr.proto.TaskRes.workload_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='task', full_name='flwr.proto.TaskRes.task', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=465,
  serialized_end=562,
)

_TASK.fields_by_name['producer'].message_type = flwr_dot_proto_dot_node__pb2._NODE
_TASK.fields_by_name['consumer'].message_type = flwr_dot_proto_dot_node__pb2._NODE
_TASK.fields_by_name['legacy_server_message'].message_type = flwr_dot_proto_dot_transport__pb2._SERVERMESSAGE
_TASK.fields_by_name['legacy_client_message'].message_type = flwr_dot_proto_dot_transport__pb2._CLIENTMESSAGE
_TASKINS.fields_by_name['task'].message_type = _TASK
_TASKRES.fields_by_name['task'].message_type = _TASK
DESCRIPTOR.message_types_by_name['Task'] = _TASK
DESCRIPTOR.message_types_by_name['TaskIns'] = _TASKINS
DESCRIPTOR.message_types_by_name['TaskRes'] = _TASKRES
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Task = _reflection.GeneratedProtocolMessageType('Task', (_message.Message,), {
  'DESCRIPTOR' : _TASK,
  '__module__' : 'flwr.proto.task_pb2'
  # @@protoc_insertion_point(class_scope:flwr.proto.Task)
  })
_sym_db.RegisterMessage(Task)

TaskIns = _reflection.GeneratedProtocolMessageType('TaskIns', (_message.Message,), {
  'DESCRIPTOR' : _TASKINS,
  '__module__' : 'flwr.proto.task_pb2'
  # @@protoc_insertion_point(class_scope:flwr.proto.TaskIns)
  })
_sym_db.RegisterMessage(TaskIns)

TaskRes = _reflection.GeneratedProtocolMessageType('TaskRes', (_message.Message,), {
  'DESCRIPTOR' : _TASKRES,
  '__module__' : 'flwr.proto.task_pb2'
  # @@protoc_insertion_point(class_scope:flwr.proto.TaskRes)
  })
_sym_db.RegisterMessage(TaskRes)


_TASK.fields_by_name['legacy_server_message']._options = None
_TASK.fields_by_name['legacy_client_message']._options = None
# @@protoc_insertion_point(module_scope)

from google.protobuf import descriptor
from google.protobuf import message
from google.protobuf import reflection
from google.protobuf import service
from google.protobuf import service_reflection
from google.protobuf import descriptor_pb2

DESCRIPTOR = descriptor.FileDescriptor(
  name='data_sharder.proto',
  package='protobuf.socketrpc')

_DATASHARDREQUEST = descriptor.Descriptor(
  name='DataShardRequest',
  full_name='protobuf.socketrpc.DataShardRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='batch_size', full_name='protobuf.socketrpc.DataShardRequest.batch_size', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
)

_DATASHARDRESPONSE = descriptor.Descriptor(
  name='DataShardResponse',
  full_name='protobuf.socketrpc.DataShardResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='batchx', full_name='protobuf.socketrpc.DataShardResponse.batchx', index=0,
      number=2, type=12, cpp_type=9, label=2,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='batchy', full_name='protobuf.socketrpc.DataShardResponse.batchy', index=0,
      number=3, type=12, cpp_type=9, label=2,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
)

DESCRIPTOR.message_types_by_name['DataShardResponse'] = _DATASHARDRESPONSE
DESCRIPTOR.message_types_by_name['DataShardRequest'] = _DATASHARDREQUEST

class DataShardResponse(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _DATASHARDRESPONSE

class DataShardRequest(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _DATASHARDREQUEST

_DATASHARDSERVICE = descriptor.ServiceDescriptor(
  name='DataShardService',
  full_name='protobuf.socketrpc.DataShardService',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=112,
  serialized_end=214,
  methods=[
  descriptor.MethodDescriptor(
    name='DataService',
    full_name='protobuf.socketrpc.DataShardService.DataService',
    index=0,
    containing_service=None,
    input_type=_DATASHARDREQUEST,
    output_type=_DATASHARDRESPONSE,
    options=None,
  ),
])

class DataShardService(service.Service):
  __metaclass__ = service_reflection.GeneratedServiceType
  DESCRIPTOR = _DATASHARDSERVICE

class DataShardService_Stub(service.Service):
  __metaclass__ = service_reflection.GeneratedServiceStubType
  DESCRIPTOR = _DATASHARDSERVICE

from models.database import Database
from models.join_type import JoinType
from models.scan_type import ScanType

JOIN_NUMBER = len(JoinType) - 1
SCAN_NUMBER = len(ScanType) - 1
AGGREGATE_FN_LIST = [
    'array_agg', 'array_agg', 'avg', 'bit_and', 'bit_or', 'bool_and', 'bool_or', 'count', 'count', 'every', 'json_agg',
    'jsonb_agg', 'json_object_agg', 'jsonb_object_agg', 'max', 'min', 'string_agg', 'sum', 'xmlagg'
]


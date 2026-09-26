# entity_id           0
# business_name       0
# business_address    0
# country             0
# dtype: int64

# entity_id                0
# business_name           46
# business_address    129408
# country                  0
# dtype: int64

# entity_id                0
# business_name           59
# business_address    136098
# country                  0
# dtype: int64

import pandas as pd
from blocker import match

source1 = pd.read_csv("student_resource/dataset/test/test_source1.tsv",sep="\t")
source2 = pd.read_csv("student_resource/dataset/test/test_source2.tsv",sep="\t")
source3 = pd.read_csv("student_resource/dataset/test/test_source3.tsv",sep="\t")

col = ['entity_id', 'business_name', 'business_address', 'country']

# mask2 = source2[col[1]].isna() | source2[col[2]].isna()
# mask3 = source3[col[1]].isna() | source3[col[2]].isna()

# print(source2[mask2])
# print(source3[mask3])

s1 = source1.iloc[0,0:]
entity_id = s1[col[0]]
name = s1["business_name"]
address = s1["business_address"]
country = s1["country"]

print(col[0],entity_id)
print("Name : ", name)
print("Address : ",address)
print("Country : ", country)

# s1_list = match(name,address,source2[:100000],source3[:100000])
# print(s1_list)

print(len(source2))
# for i in s1_list()
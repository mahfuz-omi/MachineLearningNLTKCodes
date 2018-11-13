from faker_schema.faker_schema import FakerSchema

schema = {'employee_id': 'uuid4', 'employee_name': 'name', 'employee address': 'address',
'email_address': 'email'}
faker = FakerSchema()
data = faker.generate_fake(schema)
print(data)


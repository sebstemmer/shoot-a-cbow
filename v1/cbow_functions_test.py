import cbow_functions as cbf


# get_context

output = cbf.get_context(
    3,
    [10, 11, 12, 13, 14, 15, 16],
    2
)
assert output == [11, 12, 14, 15]

output = cbf.get_context(
    1,
    [10, 11, 12, 13, 14, 15, 16],
    3
)
assert output == [10, 12, 13, 14]

output = cbf.get_context(
    4,
    [10, 11, 12, 13, 14, 15, 16],
    3
)
assert output == [11, 12, 13, 15, 16]


# create_training_data_for_sentence

output = cbf.create_training_data_for_sentence(
    [10, 11, 12, 13, 14, 15, 16],
    2
)

assert output == [
    (10, [11, 12]),
    (11, [10, 12, 13]),
    (12, [10, 11, 13, 14]),
    (13, [11, 12, 14, 15]),
    (14, [12, 13, 15, 16]),
    (15, [13, 14, 16]),
    (16, [14, 15])
]


# create_training_data

output = cbf.create_training_data(
    [[10, 11, 12, 13, 14, 15, 16], [17, 18, 19], [11]],
    2
)

assert output == [
    (10, [11, 12]),
    (11, [10, 12, 13]),
    (12, [10, 11, 13, 14]),
    (13, [11, 12, 14, 15]),
    (14, [12, 13, 15, 16]),
    (15, [13, 14, 16]),
    (16, [14, 15]),
    (17, [18, 19]),
    (18, [17, 19]),
    (19, [17, 18])
]

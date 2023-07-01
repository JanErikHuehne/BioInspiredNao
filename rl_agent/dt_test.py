import dt
import numpy as np
deci_t = dt.State_DT()

test_data = np.array([[0., 0., 1.],
                    [1, 0, 1],
                    [2, 0, 1],
                    [3, 0, 1],
                    [4, 0,1 ],
                    [4, 0, 2],
                    [3, 0 ,2],
                    [2, 0, 2],
                    [1, 0, 2 ],
                    [0, 0, 2 ],
                    ])
test_labels = np.array([[1],
                        [1],
                        [1],
                        [1],
                        [0],
                        [-1],
                        [-1],
                        [-1],
                        [-1],
                        [0],
                        ])

for t, i in zip(test_data, test_labels):
    deci_t.add_experience(t, i[0])

deci_t.update()
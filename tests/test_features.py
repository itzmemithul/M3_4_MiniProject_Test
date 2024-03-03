
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from WalmartSales_model.config.core import config
from WalmartSales_model.processing.features import OutlierHandler



def test_unemployment_variable_outlierhandler(sample_input_data):
    # Given
    encoder = OutlierHandler(variable = config.model_config.unemployment_var)
    q1, q3 = np.percentile(sample_input_data[0]['Unemployment'], q=[25, 75])
    iqr = q3 - q1
    assert sample_input_data[0].loc[5813, 'Unemployment'] > q3 + (1.5 * iqr)

    # When
    subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[5813, 'Unemployment'] <= q3 + (1.5 * iqr)


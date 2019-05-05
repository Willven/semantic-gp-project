The system is set up so that it can be imported to a Jupyter notebook (which is what was done in experiments)
or alternatively, the standalone Python file mpgsgp can be modified and run to give similar reuslts.

To view the progress of evolution, it is recommended to have tqdm installed:
	$ pip install tqdm

However, this is not a requirement.

The data used is provided in the data folder, courtesy of tennis-data.co.uk.

It is challenging to provide all the derivations of commands and results discussed in the report.
Therefore, it is recommended to refer to this for specifics as it gives the details used to run the system.
The mpgsgp.py script can then be modified to provide the desired results.

Within the data folder, there are many csv files.
reg_inputs and reg_targets were used as the training data, and are processed from the originall '2018.csv' datafile also provided.

test_input and test_targets were used to evaluate its generalisation ability.

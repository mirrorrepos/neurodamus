import pytest
import h5py
import os
import numpy as np
from pathlib import Path

SIM_DIR = Path(__file__).parent.absolute() / "simulations"

pytestmark = [
    pytest.mark.forked,
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("NEURODAMUS_NEOCORTEX_ROOT"),
        reason="Test requires loading a neocortex model to run"
    )
]


@pytest.fixture
def test_file(tmpdir):
    """
    Generates example weights file
    """
    # Define populations and their GIDs
    populations = {
        "default": [62797, 63698],
        "other_pop": [77777, 88888]
    }

    # Create a test HDF5 file with sample data
    test_file = h5py.File(tmpdir.join("test_file.h5"), 'w')

    for population, gids in populations.items():
        # Create electrodes group
        electrodes_group = test_file.create_group("electrodes/" + population)
        for gid in gids:
            # Create dataset for each GID under the electrodes group
            data = [[0.1, 0.2], [0.3, 0.4]] if gid in [62797, 77777] else [[0.5, 0.6], [0.7, 0.8]]
            electrodes_group.create_dataset(str(gid), data=data)

        # Create population group
        population_group = test_file.create_group(population)

        # Create node_ids dataset and set circuit attribute
        node_ids = population_group.create_dataset("node_ids", data=gids)
        node_ids.attrs["circuit"] = f"test_circuit_{population}.h5"

        # Create sec_ids group
        sec_ids_group = population_group.create_group("sec_ids")
        for gid in gids:
            # Create dataset for each GID under the sec_ids group
            data = [0, 1] if gid in [62797, 77777] else [1, 2]
            sec_ids_group.create_dataset(str(gid), data=data)

    yield test_file


def test_load_lfp_config(tmpdir, test_file):
    """
    Test that the 'load_lfp_config' function opens and loads correctly
    the LFP weights file and checks its format
    """
    from neurodamus.cell_distributor import LFPManager
    from neurodamus.core.configuration import ConfigurationError

    # Test loading LFP config file from invalid circuit
    lfp_invalid = LFPManager()
    lfp_weights_file = tmpdir.join("test_file.h5")
    pop_circuit_invalid = {"default": "invalid_circuit.h5"}
    lfp_invalid.load_lfp_config(lfp_weights_file, pop_circuit_invalid)
    # File is closed
    assert not lfp_invalid._lfp_file

    # Create an instance of the class
    lfp = LFPManager()
    pop_circuit_dict = {"wrong_pop": "test_circuit2.h5", "default": "test_circuit_default.h5"}

    # Test loading LFP configuration from file
    lfp.load_lfp_config(lfp_weights_file, pop_circuit_dict)
    assert lfp._lfp_file
    assert isinstance(lfp._lfp_file, h5py.File)
    assert "/electrodes/default" in lfp._lfp_file
    assert "/default/node_ids" in lfp._lfp_file
    assert "/default/sec_ids" in lfp._lfp_file
    assert lfp._lfp_file["default"]["node_ids"].attrs['circuit'] == "test_circuit_default.h5"

    # Test loading LFP configuration from file with wrong format
    del lfp._lfp_file["default"]["node_ids"].attrs['circuit']
    with pytest.raises(ConfigurationError):
        lfp.load_lfp_config(lfp_weights_file, pop_circuit_dict)

    del lfp._lfp_file["default"]["node_ids"]
    with pytest.raises(ConfigurationError):
        lfp.load_lfp_config(lfp_weights_file, pop_circuit_dict)

    # Test loading LFP configuration from invalid file
    lfp_weights_invalid_file = "./invalid_file.h5"
    with pytest.raises(ConfigurationError):
        lfp.load_lfp_config(lfp_weights_invalid_file, pop_circuit_dict)


def test_read_lfp_factors(test_file):
    """
    Test that the 'read_lfp_factors' function correctly extracts the LFP factors
    for the specified gid and section ids from the weights file
    """
    from neurodamus.cell_distributor import LFPManager
    # Create an instance of the class
    lfp = LFPManager()
    lfp._lfp_file = test_file
    # Test the function with valid input (node_id is 0 based, so expected 62797 in the file)
    gid = 62798
    section_ids = [1, 2]
    result = lfp.read_lfp_factors(gid, section_ids).to_python()
    expected_result = [0.3, 0.4]
    assert result == expected_result, f'Expected {expected_result}, but got {result}'

    # Test the function with invalid input (non-existent gid)
    gid = 2
    section_ids = [0, 1]
    result = lfp.read_lfp_factors(gid, section_ids).to_python()
    expected_result = []
    assert result == expected_result, f'Expected {expected_result}, but got {result}'


def test_number_electrodes(test_file):
    """
    Test that the 'get_number_electrodes' function correctly extracts the number of
    electrodes in the weights file for a certain gid
    """
    from neurodamus.cell_distributor import LFPManager
    # Create an instance of the class
    lfp = LFPManager()
    lfp._lfp_file = test_file
    # Test the function with valid input
    gid = 62798
    result = lfp.get_number_electrodes(gid)
    expected_result = 2
    assert result == expected_result, f'Expected {expected_result}, but got {result}'

    # Test the function with invalid input (non-existent gid)
    gid = 2
    result = lfp.get_number_electrodes(gid)
    expected_result = 0
    assert result == expected_result, f'Expected {expected_result}, but got {result}'


def _create_tmpconfig_lfp(config_file, lfp_file):
    import fileinput
    import shutil
    from tempfile import NamedTemporaryFile

    lfp_replace = "\"electrodes_file\": \"" + str(lfp_file) + "\""
    suffix = ".json" if config_file.endswith(".json") else ".BC"
    tmp_file = NamedTemporaryFile(suffix=suffix, dir=os.path.dirname(config_file), delete=True)
    shutil.copy2(config_file, tmp_file.name)

    with fileinput.FileInput(tmp_file.name, inplace=True) as file:
        for line in file:
            if config_file.endswith(".json"):
                print(line.replace("\"electrodes_file\": \"electrodes_file.h5\"",
                                   lfp_replace), end='')
    return tmp_file


def _read_sonata_lfp_file(lfp_file):
    import libsonata
    report = libsonata.ElementReportReader(lfp_file)
    pop_name = report.get_population_names()[0]
    node_ids = report[pop_name].get_node_ids()
    data = report[pop_name].get()
    return node_ids, data


def test_v5_sonata_lfp(tmpdir, test_file):
    import numpy.testing as npt
    from neurodamus import Neurodamus

    config_file = str(SIM_DIR / "v5_sonata" / "simulation_config_lfp.json")
    output_dir = str(SIM_DIR / "v5_sonata" / "output_coreneuron")
    population_name = "default"

    test_file[population_name]["node_ids"].attrs['circuit'] = "/gpfs/bbp.cscs.ch/project/proj1/" \
                                             "circuits/SomatosensoryCxS1-v5.r0/O1-sonata/sonata" \
                                             "/networks/nodes/default/nodes.h5"
    lfp_weights_file = tmpdir.join("test_file.h5")
    tmp_file = _create_tmpconfig_lfp(config_file, lfp_weights_file)

    nd = Neurodamus(tmp_file.name, output_path=output_dir)
    nd.run()

    # compare results with refs
    t3_data = np.array([-7.392764091491699219e-04, -1.529844943434000015e-03,
                        9.888911154121160507e-04, 1.153199234977364540e-03])
    t7_data = np.array([-7.781040039844810963e-04, -1.588534913025796413e-03,
                        7.941523799672722816e-04, 9.259916841983795166e-04])
    node_ids = np.array([62797, 63698])
    result_ids, result_data = _read_sonata_lfp_file(os.path.join(output_dir, "lfp.h5"))

    npt.assert_allclose(result_data.data[3], t3_data)
    npt.assert_allclose(result_data.data[7], t7_data)
    npt.assert_allclose(result_ids, node_ids)

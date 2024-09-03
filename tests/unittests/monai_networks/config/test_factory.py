from clinicadl.monai_networks.config import ImplementedNetworks, create_network_config


def test_create_training_config():
    for network in [e.value for e in ImplementedNetworks]:
        create_network_config(network)

    config_class = create_network_config("DenseNet")
    config = config_class(
        spatial_dims=1,
        in_channels=2,
        out_channels=3,
    )
    assert config.network == "DenseNet"
    assert config.spatial_dims == 1
    assert config.in_channels == 2
    assert config.out_channels == 3
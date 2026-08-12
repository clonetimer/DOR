from fec.extensions.catalog import assess_extension_readiness

def test_catalog_blocks_extensions_without_semantic_preconditions():
    r=assess_extension_readiness({'external_group_key':True,'trusted_normal_per_group':True,'matched_G_A_topology':True})
    assert r['mondrian_calibration']['ready']
    assert not r['weighted_conformal']['ready']
    assert not r['opinion_fusion']['ready']

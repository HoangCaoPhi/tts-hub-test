from . import pocket, vieneu

PROVIDERS = {
    "pocket": pocket.synthesize,
    "vieneu": vieneu.synthesize,
}

PROVIDER_INFO = {
    "pocket": pocket.info,
    "vieneu": vieneu.info,
}

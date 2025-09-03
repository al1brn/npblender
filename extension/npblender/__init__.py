from . import npblender as _pkg

def register():
    _pkg.enginemod.register()

def unregister():
    _pkg.enginemod.unregister()

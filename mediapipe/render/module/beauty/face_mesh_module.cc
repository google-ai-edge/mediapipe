#include "face_mesh_module.h"
#include "face_mesh_module_imp.h"


namespace Opipe {
    

    FaceMeshModule::~FaceMeshModule() {
        
    }

    FaceMeshModule::FaceMeshModule() {
        
    }

    FaceMeshModule* FaceMeshModule::create() {
        FaceMeshModuleIMP *instance = new FaceMeshModuleIMP();
        return instance;
    }

   
}

import requests
import json
from mp_api.client import MPRester
from emmet.core.summary import HasProps

API_KEY = "e3wGuqHhW3MOckChl0fXia3Rvy1Yahoa"
URL = "https://materialsproject.org/rest/v2/materials/mp-1"
URL2 = "https://materialsproject.org/rest/v2/materials"
URL3 = "https://materialsproject.org/rest/v2/query"

def download_materials_data(output_file):
    try:
        with MPRester(API_KEY) as mpr:
            docs = mpr.summary.search(
                                    # elements=["Si"], 
                                    exclude_elements=["C"], # Se filtra por materiales inorgánicos (no tienen átomos de Carbono)
                                    energy_above_hull=[0, 0.1], # Se filtra por energía de hull positiva (se filtran aproximadamente por materiales en estado sólido)
                                    density=[1, 10], # Se filtra por densidad de masa entre 1 y 10 gramos por centímetro cúbico
                                    formation_energy=[-10, 10], # Se filtra por energía de formación de átomo (-10 a 10 eV/átomo)
                                    band_gap=[0.5, 1.0], # Se filtra por brecha de banda (0.5 y 1.0 eV -> se consideran semiconductores)
                                    has_props=[HasProps.electronic_structure, HasProps.thermo],
                                    deprecated=False, # Se filtran los materiales obsoletos
                                    # is_stable=True, # Se filtra por materiales estables (así los VAE y GAN generarán estructuras más estables y aplicables en la vida real)
                                    # is_metal=False, # Este filtro no es necesario puesto que con el resto de filtros ninguno de los materiales es metálico
                                    # fields=["material_id", "formula_pretty", "band_gap", "density"]
                                    all_fields=True,
                                    )
            materials_data = {doc.material_id: {
                                        # Composición química
                                        "formula_pretty": doc.formula_pretty, 
                                        "elements": [e.symbol for e in doc.elements],
                                        "nelements": doc.nelements,

                                        # Propiedades físicas
                                        "band_gap": doc.band_gap, 
                                        "volume": doc.volume,
                                        "density": doc.density,
                                        "total_magnetization": doc.total_magnetization,
                                        "energy_per_atom": doc.energy_per_atom,
                                        "energy_above_hull": doc.energy_above_hull,
                                        "nsites": doc.nsites,
                                        "num_magnetic_sites": doc.num_magnetic_sites,

                                        # Datos de estructura cristalina (necesarios para la generación de imágenes)
                                        # Los más importantes para visualización básica son: lattice, species y cart_coords
                                        "structure_lattice": doc.structure.lattice.matrix.tolist(),
                                        "structure_species": [str(specie) for specie in doc.structure.species],
                                        "structure_cart_coords": doc.structure.cart_coords.tolist(),
                                        "structure_charge": doc.structure.charge,
                                        # "structure_validate_proximity": doc.structure.validate_proximity,
                                        # "structure_to_unit_cell": doc.structure.to_unit_cell,
                                        "structure_site_properties": doc.structure.site_properties,

                                        # Simetría
                                        "symmetry_crystal_system": doc.symmetry.crystal_system.value if doc.symmetry.crystal_system else None,
                                        "symmetry_symbol": doc.symmetry.symbol,
                                        "symmetry_number": doc.symmetry.number,
                                        "symmetry_point_group": doc.symmetry.point_group,
                                        "symmetry_symprec": doc.symmetry.symprec,
                                        "symmetry_version": doc.symmetry.version,

                                        # Propiedades termodinámicas
                                        # "thermo.energy": doc.has_props.thermo.energy,
                                        # "thermo.enthalpy": doc.thermo.enthalpy,
                                        # "thermo.entropy": doc.thermo.entropy,
                                        # "thermo.heat_capacity": doc.thermo.heat_capacity,

                                        # Propiedades elásticas
                                        # No se añaden porque en la gran mayoría de casos son null (probablemente por los filtros)
                                        # "k_voigt": doc.k_voigt,
                                        # "k_reuss": doc.k_reuss,
                                        # "k_vrh": doc.k_vrh,
                                        # "g_voigt": doc.g_voigt,
                                        # "g_reuss": doc.g_reuss,
                                        # "g_vrh": doc.g_vrh,

                                        # Otros
                                        "efermi": doc.efermi,
                                        }
                                for doc in docs}
        if materials_data:
            with open(output_file, "w") as f:
                json.dump(materials_data, f)
            print("Datos descargados y guardados en", output_file)
        else:
            print("Error al descargar los datos:") #, docs.status_code
    except Exception as e:
        print("Se produjo un error al descargar los datos", str(e))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Descargar y guardar datos del Materials Project")
    parser.add_argument("--output_file", default="materials_data.json", help="Archivo de salida para guardar los datos")

    args = parser.parse_args() 

    download_materials_data(args.output_file)

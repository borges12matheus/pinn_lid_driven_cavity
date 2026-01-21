import pyvista as pv
import pandas as pd


def create_df_cavity(mesh_path: str, df_output: str) -> None:
    mesh = pv.read(mesh_path)
    # junta todos os blocos
    mesh = mesh.combine()

    pts = mesh.points
    U = mesh.point_data["U"]
    p = mesh.point_data["p"]

    df = pd.DataFrame({
        "x": pts[:,0],
        "y": pts[:,1],
        "u": U[:,0],
        "v": U[:,1],
        "p": p
    })

    df.to_csv(df_output, index=False)

if __name__ == "__main__":
    create_df_cavity("CFD/cavity_coarse/VTK/cavity_coarse_100.vtm", "data/cavity_20_full.csv")
    print("Create cavity 20 done")
    create_df_cavity("CFD/cavity_fine/VTK/cavity_fine_80.vtm", "data/cavity_40_full.csv")
    print("Create cavity 40 done")
    create_df_cavity("CFD/cavity_ref/VTK/cavityRef_80.vtm", "data/cavity_80_full.csv")
    print("Create cavity 80 done")
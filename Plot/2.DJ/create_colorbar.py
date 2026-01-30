import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

# Usage: python create_colorbar.py --vmax 100 --save colorbar.pdf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vmax", type=float, required=True, help="The fixed vmax used in your plots")
    parser.add_argument("--save", type=str, default="colorbar.pdf")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = plt.cm.seismic
    norm = mpl.colors.Normalize(vmin=-args.vmax, vmax=args.vmax)

    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax, 
                      orientation='horizontal', 
                      extend='both')            

    cb.set_label(r'$\omega$', fontsize=14)
    cb.ax.tick_params(labelsize=12)

    plt.savefig(args.save, bbox_inches='tight', dpi=300)
    print(f"Saved horizontal colorbar to {args.save} with range [-{args.vmax}, {args.vmax}]")

if __name__ == "__main__":
    main()
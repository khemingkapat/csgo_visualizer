{
  description = "CS:GO Visualizer Based on Streamlit and Python";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python311;

        pythonEnv = python.withPackages (
          ps: with ps; [
            pip
            setuptools
            wheel
            numpy
            pandas
            matplotlib
            plotly
            seaborn
            scikit-learn
            fastparquet
            pyarrow
            streamlit
            igraph
            leidenalg
          ]
        );

      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            jupyter
            pandoc
            texlive.combined.scheme-full
          ];
          buildInputs = [
            pythonEnv
            pkgs.nodejs
          ];

          shellHook = ''
            echo "Activating Python"

            # Ensure Jupyter uses the correct Python
            export PATH="${pythonEnv}/bin:$PATH"


            echo "Python Registered in PATH"
          '';
        };
      }
    );
}

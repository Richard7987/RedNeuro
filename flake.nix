{
  description = "Red neuronal from scratch con NumPy para HTRU2";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python3.withPackages (ps: with ps; [
        numpy
        pandas
        matplotlib
        scikit-learn
        jupyter
        pytest
      ]);
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ python ];
        shellHook = ''
          export PYTHONPATH="$PWD/src:$PYTHONPATH"
          echo "Entorno Redneuro listo: $(python --version)"
        '';
      };
    };
}

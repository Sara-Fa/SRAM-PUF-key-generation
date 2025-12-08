import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .formulas import delta_from_K_eta, p_unreliable_exact, find_minimum_surplus


@dataclass
class UnreliableResult:
    K: int
    eta: float
    delta: float
    lambda_used: float
    p_unreliable_exact: float
    L: int | None = None
    P_disc: float | None = None
    G_min: int | None = None
    discard_prob_at_G: float | None = None


def load_lambda_estimates(lambda_json_path: pathlib.Path) -> Dict[str, dict]:
    with open(lambda_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_p_unreliable_for_grid(lambda_json_path: pathlib.Path,
                                  output_dir: pathlib.Path,
                                  k_list: List[int],
                                  eta_param: float,
                                  use_lambda_key: str = 'lambda_window',
                                  L_required: int | None = None,
                                  P_disc: float | None = None,
                                  G_max: int = 1000) -> Dict[str, List[UnreliableResult]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lambda_map = load_lambda_estimates(lambda_json_path)

    all_results: Dict[str, List[UnreliableResult]] = {}
    for chip_id, payload in lambda_map.items():
        lambda_val = payload.get(use_lambda_key)
        if lambda_val is None:
            # Fallback to window if requested key missing
            lambda_val = payload.get('lambda_window')

        chip_results: List[UnreliableResult] = []
        for K in k_list:
            delta = float(delta_from_K_eta(int(K), eta_param))
            p_exact = float(p_unreliable_exact(delta, lambda_val))
            result_kwargs = dict(
                K=int(K), eta=float(eta_param), delta=float(delta), lambda_used=float(lambda_val),
                p_unreliable_exact=p_exact
            )
            if L_required is not None and P_disc is not None:
                G_min, prob_at_G = find_minimum_surplus(int(L_required), float(p_exact), float(P_disc), int(G_max))
                result_kwargs.update({
                    'L': int(L_required),
                    'P_disc': float(P_disc),
                    'G_min': None if G_min is None else int(G_min),
                    'discard_prob_at_G': None if prob_at_G is None else float(prob_at_G),
                })
            chip_results.append(UnreliableResult(**result_kwargs))

        all_results[chip_id] = chip_results

    # Save JSON and CSV
    json_path = output_dir / 'p_unreliable_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            chip: [r.__dict__ for r in res]
            for chip, res in all_results.items()
        }, f, indent=2)

    # Flatten CSV-like output
    csv_path = output_dir / 'p_unreliable_results.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('chip_id,K,eta,delta,lambda_used,p_unreliable_exact,L,P_disc,G_min,discard_prob_at_G\n')
        for chip_id, res_list in all_results.items():
            for r in res_list:
                f.write(f"{chip_id},{r.K},{r.eta},{r.delta},{r.lambda_used},{r.p_unreliable_exact},{r.L},{r.P_disc},{r.G_min},{r.discard_prob_at_G}\n")

    return all_results


if __name__ == '__main__':
    base_dir = pathlib.Path(__file__).parent
    lambda_json = base_dir / 'results' / 'lambda_estimates.json'
    out_dir = base_dir / 'results'

    # Example: K grid and eta specified by user
    k_grid = list(np.unique(np.r_[10, 20, 50, 100, 200, 500, 1000]))
    eta_target = 1.0 - 1e-9
    compute_p_unreliable_for_grid(lambda_json, out_dir, k_grid, eta_target, use_lambda_key='lambda_window', L_required=128, P_disc=1e-9, G_max=1e7)



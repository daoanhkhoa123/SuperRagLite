def run_single(pipe, run_dict):
  result = pipe.run(run_dict)

  for i in range(8):  # at max 16 tries
      if "regenerate" not in result["hallu_router"]:
        break

      print("Checking at iteration", i)
      result = pipe.run(run_dict)
  else:
    return result

  return result["hallu_router"]["pass_answer"]


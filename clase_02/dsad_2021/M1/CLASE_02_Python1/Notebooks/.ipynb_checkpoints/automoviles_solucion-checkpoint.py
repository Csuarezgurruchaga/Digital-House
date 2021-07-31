def make_car(fabricante, modelo, **info):
    result = {'fabricante': fabricante,
              'modelo': modelo}
    for key in info.keys():
        result[key] = info[key]
    return result    
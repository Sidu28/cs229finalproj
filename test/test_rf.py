from random_forest import RFPredictor


def test_rf():
    rf = RFPredictor('data/dummy.csv')
    rmse = rf.run()

    avg = sum(rmse)/len(rmse)
    assert(avg < 1)


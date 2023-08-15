import os, sys
sys.path.append("../lab1")
sys.path.append("../lab2")
# print(sys.path)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from lab1.learn_from_query import CardinalityEstimator
from lab2.cost_learning import CostEstimator


host = ('localhost', 8888)

cost_estimator = CostEstimator()
caridinality_estimator = CardinalityEstimator()

class Resquest(BaseHTTPRequestHandler):
    def __int__(self):
        super(Resquest, self).__int__()


    def handle_cardinality_estimate(self, req_data):
        req_data = req_data.decode()
        est_caridinality = caridinality_estimator.get_cardinality(req_data)
        print("cardinality_estimate post_data: " + str(req_data) + '\n est caridinality: {}'.format(est_caridinality))
        return {"selectivity": est_caridinality, "err_msg": ""} # return the selectivity

    def handle_cost_estimate(self, req_data):
        req_data = json.loads(eval(str(req_data)))
        est_cost = cost_estimator.get_cost(req_data)
        print("cost_estimate post_data: " + str(req_data) + '\nest cost: {}'.format(est_cost))
        return {"cost": est_cost, "err_msg": ""} # return the cost

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        req_data = self.rfile.read(content_length)
        resp_data = ""
        if self.path == "/cardinality":
            resp_data = self.handle_cardinality_estimate(req_data)
        elif self.path == "/cost":
            resp_data = self.handle_cost_estimate(req_data)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(resp_data).encode())


if __name__ == '__main__':
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()

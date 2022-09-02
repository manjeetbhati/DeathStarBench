#include <signal.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TServerSocket.h>

#include "../utils.h"
#include "../utils_thrift.h"
#include "MediaHandler.h"

using apache::thrift::protocol::TBinaryProtocolFactory;
using apache::thrift::server::TThreadedServer;
using apache::thrift::transport::TFramedTransportFactory;
using apache::thrift::transport::TServerSocket;
using namespace social_network;

void sigintHandler(int sig) { exit(EXIT_SUCCESS); }

int main(int argc, char *argv[]) {
  signal(SIGINT, sigintHandler);
  init_logger();
  SetUpTracer("config/jaeger-config.yml", "media-service");
  json config_json;
  if (load_config_file("config/service-config.json", &config_json) != 0) {
    exit(EXIT_FAILURE);
  }

  int port = config_json["media-service"]["port"];

  std::string media_filter_addr = config_json["media-filter-service"]["addr"];
  int media_filter_port = config_json["media-filter-service"]["port"];
  int media_filter_conns = config_json["media-filter-service"]["connections"];
  int media_filter_timeout = config_json["media-filter-service"]["timeout_ms"];
  int media_filter_keepalive =
      config_json["media-filter-service"]["keepalive_ms"];

  ClientPool<ThriftClient<MediaFilterServiceClient>> media_filter_client_pool(
      "media-filter-service", media_filter_addr, media_filter_port, 0,
      media_filter_conns, media_filter_timeout, media_filter_keepalive,
      config_json);

  std::shared_ptr<TServerSocket> server_socket =
      get_server_socket(config_json, "0.0.0.0", port);
  TThreadedServer server(
      std::make_shared<MediaServiceProcessor>(
          std::make_shared<MediaHandler>(&media_filter_client_pool)),
      server_socket, std::make_shared<TFramedTransportFactory>(),
      std::make_shared<TBinaryProtocolFactory>());

  LOG(info) << "Starting the media-service server...";
  server.serve();
}

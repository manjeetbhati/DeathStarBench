#ifndef SOCIAL_NETWORK_MICROSERVICES_SRC_MEDIASERVICE_MEDIAHANDLER_H_
#define SOCIAL_NETWORK_MICROSERVICES_SRC_MEDIASERVICE_MEDIAHANDLER_H_

#include <chrono>
#include <future>
#include <iostream>
#include <regex>
#include <string>

#include "../../gen-cpp/MediaFilterService.h"
#include "../../gen-cpp/MediaService.h"
#include "../ClientPool.h"
#include "../ThriftClient.h"
#include "../base64.h"
#include "../gzip.h"
#include "../logger.h"
#include "../tracing.h"

// 2018-01-01 00:00:00 UTC
#define CUSTOM_EPOCH 1514764800000

namespace social_network {

class MediaHandler : public MediaServiceIf {
 public:
  MediaHandler(ClientPool<ThriftClient<MediaFilterServiceClient>> *);
  ~MediaHandler() override = default;

  void ComposeMedia(std::vector<Media> &_return, int64_t,
                    const std::vector<int64_t> &,
                    const std::vector<std::string> &,
                    const std::vector<std::string> &,
                    const std::map<std::string, std::string> &) override;

 private:
  ClientPool<ThriftClient<MediaFilterServiceClient>> *_media_filter_client_pool;
};

MediaHandler::MediaHandler(ClientPool<ThriftClient<MediaFilterServiceClient>>
                               *media_filter_client_pool) {
  _media_filter_client_pool = media_filter_client_pool;
}

void MediaHandler::ComposeMedia(
    std::vector<Media> &_return, int64_t req_id,
    const std::vector<int64_t> &media_ids,
    const std::vector<std::string> &media_types,
    const std::vector<std::string> &media_data_list,
    const std::map<std::string, std::string> &carrier) {
  // Initialize a span
  TextMapReader reader(carrier);
  std::map<std::string, std::string> writer_text_map;
  TextMapWriter writer(writer_text_map);
  auto parent_span = opentracing::Tracer::Global()->Extract(reader);
  auto span = opentracing::Tracer::Global()->StartSpan(
      "compose_media_server", {opentracing::ChildOf(parent_span->get())});
  opentracing::Tracer::Global()->Inject(span->context(), writer);

  if (media_types.size() != media_ids.size()) {
    ServiceException se;
    se.errorCode = ErrorCode::SE_THRIFT_HANDLER_ERROR;
    se.message =
        "The lengths of media_id list and media_type list are not equal";
    throw se;
  }

  // media-filter-service
  auto media_filter_future = std::async(std::launch::async, [&]() {
    auto media_filter_span = opentracing::Tracer::Global()->StartSpan(
        "media_filter_client", {opentracing::ChildOf(&span->context())});

    std::map<std::string, std::string> media_filter_writer_text_map;
    TextMapWriter media_filter_writer(media_filter_writer_text_map);
    opentracing::Tracer::Global()->Inject(media_filter_span->context(),
                                          media_filter_writer);

    auto media_filter_client_wrapper = _media_filter_client_pool->Pop();
    if (!media_filter_client_wrapper) {
      ServiceException se;
      se.errorCode = ErrorCode::SE_THRIFT_CONN_ERROR;
      se.message = "Failed to connected to media-filter-service";
      throw se;
    }
    std::vector<bool> _return_media_filter;
    auto media_filter_client = media_filter_client_wrapper->GetClient();
    try {
      media_filter_client->MediaFilter(_return_media_filter, req_id, media_ids,
                                       media_types, media_data_list,
                                       media_filter_writer_text_map);
    } catch (...) {
      LOG(error) << "Failed to upload medias to media-filter-service";
      _media_filter_client_pool->Remove(media_filter_client_wrapper);
      throw;
    }

    _media_filter_client_pool->Keepalive(media_filter_client_wrapper);
    return _return_media_filter;
  });

  std::vector<bool> media_filter;
  try {
    media_filter = media_filter_future.get();
  } catch (...) {
    LOG(error) << "Failed to get media_filter from media-filter-service";
    throw;
  }

  /********** debug ***********/
  std::string debug_str = "media_filter: ";
  for (int i = 0; i < media_filter.size(); ++i) {
    if (media_filter[i])
      debug_str += " true; ";
    else
      debug_str += " false; ";
  }
  if (media_filter.size() > 0) LOG(info) << debug_str;
  /**************/

  for (int i = 0; i < media_ids.size(); ++i) {
    // nsfw images are removed
    if (!media_filter[i]) continue;

    Media new_media;
    new_media.media_id = media_ids[i];
    new_media.media_type = media_types[i];
    // compress images
    try {
      // LOG(info) << "original media.data: " << media_data_list[i];
      std::string compressed_media_data =
          Gzip::compress(Base64::decode(media_data_list[i]));
      new_media.media_data =
          Base64::encode(reinterpret_cast<const unsigned char *>(
                             compressed_media_data.c_str()),
                         compressed_media_data.length());
      // LOG(info) << "compressed media.data: " << new_media.media_data;
    } catch (...) {
      LOG(error) << "Failed to compress images";
      throw;
    }
    _return.emplace_back(new_media);
  }

  span->Finish();
}

}  // namespace social_network

#endif  // SOCIAL_NETWORK_MICROSERVICES_SRC_MEDIASERVICE_MEDIAHANDLER_H_

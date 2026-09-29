package com.example.llama

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.llama.data.ChatMessage
import io.noties.markwon.Markwon
import io.noties.markwon.ext.tables.TablePlugin

typealias Message = ChatMessage

class MessageAdapter(
    private val messages: List<ChatMessage>,
    context: Context
) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    private val markwon: Markwon = Markwon.builder(context)
        .usePlugin(TablePlugin.create(context))
        .build()

    companion object {
        private const val VIEW_TYPE_USER = 1
        private const val VIEW_TYPE_ASSISTANT = 2
        const val PAYLOAD_STREAMING = "PAYLOAD_STREAMING"
    }

    override fun getItemViewType(position: Int): Int {
        return if (messages[position].isUser) VIEW_TYPE_USER else VIEW_TYPE_ASSISTANT
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val layoutInflater = LayoutInflater.from(parent.context)
        return if (viewType == VIEW_TYPE_USER) {
            val view = layoutInflater.inflate(R.layout.item_message_user, parent, false)
            UserMessageViewHolder(view)
        } else {
            val view = layoutInflater.inflate(R.layout.item_message_assistant, parent, false)
            AssistantMessageViewHolder(view)
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int, payloads: MutableList<Any>) {
        if (payloads.contains(PAYLOAD_STREAMING)) {
            val message = messages[position]
            val textView = holder.itemView.findViewById<TextView>(R.id.msg_content)
            textView.text = message.content
            return
        }
        super.onBindViewHolder(holder, position, payloads)
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val message = messages[position]
        val textView = holder.itemView.findViewById<TextView>(R.id.msg_content)
        if (message.isUser) {
            textView.text = message.content
            val ivImage = holder.itemView.findViewById<android.widget.ImageView?>(R.id.iv_attached_image)
            if (ivImage != null) {
                if (message.imageUri != null) {
                    ivImage.visibility = View.VISIBLE
                    try {
                        val uri = android.net.Uri.parse(message.imageUri)
                        ivImage.setImageURI(uri)
                    } catch (e: Exception) {
                        ivImage.visibility = View.GONE
                    }
                } else {
                    ivImage.visibility = View.GONE
                }
            }
        } else {
            // Render markdown syntax (headers, code blocks, lists, bold, tables)
            markwon.setMarkdown(textView, message.content)
        }
    }

    override fun getItemCount(): Int = messages.size

    class UserMessageViewHolder(view: View) : RecyclerView.ViewHolder(view)
    class AssistantMessageViewHolder(view: View) : RecyclerView.ViewHolder(view)
}
